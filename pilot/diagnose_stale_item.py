"""Rebuild one STALE scenario's graph from the LLM cache (free) and
report: graph size, whether facts near the relevant sessions were
captured, and what retrieval returns for each probing query. Companion
to diagnose_question.py / diagnose_probe_item.py.

Usage: uv run python pilot/diagnose_stale_item.py <uid prefix>
"""

import json
import sys

from bte.bench import render_session
from bte.conflict import (ConflictDetector, DomainDependencies,
                          make_llm_adjudicator, make_llm_proposer)
from bte.extraction import extract_facts
from bte.ingest import Ingestor
from bte.llm import CachedLLM
from bte.memory import BJGMemory
from bte.retrieval import CachedEmbedder, Retriever
from bte.stale_bench import STALE_READER_SYSTEM

DATA = "data/stale/T1_T2_400_FULL.json"


def main():
    prefix = sys.argv[1]
    data = json.load(open(DATA))
    scenario = next(s for s in data if s["uid"].startswith(prefix))

    print(f"uid: {scenario['uid']}")
    print(f"M_old: {scenario['M_old']}")
    print(f"M_new: {scenario['M_new']}")
    print(f"explanation: {scenario['explanation'][:200]}")
    print(f"relevant_session_index: {scenario['relevant_session_index']}")

    extractor = CachedLLM(model="deepseek/deepseek-v3.2",
                          base_url="https://openrouter.ai/api/v1",
                          api_key_env="OPENROUTER_API_KEY",
                          extra={"max_tokens": 4000})
    reader = CachedLLM(model="qwen/qwen3.5-9b",
                       base_url="https://openrouter.ai/api/v1",
                       api_key_env="OPENROUTER_API_KEY")
    embedder = CachedEmbedder()
    ing = Ingestor(
        extract=lambda text, ts, ctx: extract_facts(extractor, text, ts, ctx),
        detector=ConflictDetector(adjudicate=make_llm_adjudicator(extractor),
                                  domain_deps=DomainDependencies(),
                                  propose=make_llm_proposer(extractor)),
    )
    mem = BJGMemory(ing, Retriever(ing.graph, embedder),
                    reader=reader.complete_text,
                    reader_system=STALE_READER_SYSTEM)

    relevant = set(scenario["relevant_session_index"])
    for i, (session, ts) in enumerate(
            zip(scenario["haystack_session"], scenario["timestamps"])):
        text = render_session(session)
        before = len(ing.graph.edges)
        try:
            mem.ingest_session([text], ts)
        except Exception as e:
            print(f"  session {i}: INGEST FAILED {type(e).__name__}: {e}")
            continue
        added = len(ing.graph.edges) - before
        marker = " <== RELEVANT" if i in relevant else ""
        if i in relevant or added == 0:
            print(f"  session {i} ({ts}): +{added} edges{marker}")

    print(f"\ntotal graph size: {len(ing.graph.edges)} edges, "
          f"{sum(1 for e in ing.graph.edges if ing.graph.is_active(e))} active")

    for via in ("semantic", "domain"):
        checks = [d for d in ing.detector.log if d.via == via]
        if not checks and via == "semantic":
            continue
        print(f"\n{via} conflicts found: "
              f"{sum(1 for d in checks if d.conflict)} "
              f"(logged decisions: {len(checks)})")
        for d in checks:
            if d.conflict:
                new_e = ing.graph.edges[d.new_edge_id]
                old_e = ing.graph.edges[d.old_edge_id]
                print(f"    CONFLICT axis={d.axis}: "
                      f"({new_e.subject},{new_e.relation},{new_e.object}) "
                      f"[{new_e.domain}] vs "
                      f"({old_e.subject},{old_e.relation},{old_e.object}) "
                      f"[{old_e.domain}]")

    deps = ing.detector.domain_deps
    if deps is not None and deps.learned:
        print("\nlearned domain dependencies (pair: confirmed conflicts):")
        for (a, b), n in sorted(deps.learned.items(), key=lambda kv: -kv[1]):
            marker = "" if b in deps.prior.get(a, ()) else "  <== beyond prior"
            print(f"    {a} -> {b}: {n}{marker}")

    print("\n--- edges from relevant sessions (by t_transaction match) ---")
    relevant_ts = {scenario["timestamps"][i] for i in relevant}
    for eid, e in ing.graph.edges.items():
        if e.t_transaction in relevant_ts:
            flag = "ACTIVE" if ing.graph.is_active(eid) else "superseded"
            print(f"  ({e.subject} | {e.relation} | {e.object}) "
                  f"[{e.source_type}] {flag} t={e.t_transaction}")

    ref_time = scenario["timestamps"][-1]
    for dim, qkey in [("dim1", "dim1_query"), ("dim2", "dim2_query"),
                      ("dim3", "dim3_query")]:
        query = scenario["probing_queries"][qkey]
        print(f"\n--- {dim}: {query}")
        hits = mem.retriever.retrieve(query, reference_time=ref_time, k=8)
        for e in hits:
            print(f"    ({e.subject} | {e.relation} | {e.object})")
        if not hits:
            print("    NONE retrieved")
        answer = mem.answer(query, reference_time=ref_time)
        print(f"    answer: {answer[:200]}")


if __name__ == "__main__":
    main()
