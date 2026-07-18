"""Rebuild one probe item's graph from the LLM cache (free) and report
the internal state: what got extracted per turn, conflict decisions,
BBP propagations, and final active/superseded edges. Companion to
diagnose_question.py for LongMemEval.

Usage: uv run python pilot/diagnose_probe_item.py <probe_id>
"""

import sys

from bte.conflict import ConflictDetector, make_llm_adjudicator
from bte.extraction import extract_facts
from bte.ingest import Ingestor
from bte.llm import CachedLLM
from bte.memory import BJGMemory
from bte.probe import DOMAINS, domain_rules, generate
from bte.retrieval import CachedEmbedder, Retriever

DOMAIN_BY_NAME = {d.name: d for d in DOMAINS}


def main():
    probe_id = sys.argv[1]
    item = next(i for i in generate() if i.probe_id == probe_id)
    d = DOMAIN_BY_NAME[item.domain]

    print(f"probe_id: {item.probe_id}")
    print(f"domain: {item.domain} | depth: {item.hop_depth} | "
          f"contradicted: {item.contradicted} | axis: {item.axis} | "
          f"density: {item.density}")
    print(f"question: {item.question}")
    print(f"gold_pre: {item.gold_pre!r} | gold_post: {item.gold_post!r}")

    extractor = CachedLLM(model="deepseek/deepseek-v3.2",
                          base_url="https://openrouter.ai/api/v1",
                          api_key_env="OPENROUTER_API_KEY",
                          extra={"max_tokens": 4000})
    reader = CachedLLM(model="qwen/qwen3.5-9b",
                       base_url="https://openrouter.ai/api/v1",
                       api_key_env="OPENROUTER_API_KEY")
    detector = ConflictDetector(adjudicate=make_llm_adjudicator(extractor))
    vocab = list(d.relations) + list(d.conclusions)
    ing = Ingestor(
        extract=lambda text, ts, ctx: extract_facts(
            extractor, text, ts, ctx, relation_vocab=vocab),
        detector=detector,
        rules=domain_rules(d),
    )
    mem = BJGMemory(ing, Retriever(ing.graph, CachedEmbedder()),
                    reader=reader.complete_text)

    n = len(item.sessions)
    for i, session in enumerate(item.sessions):
        text = " ".join(session)
        marker = " <== CONTRADICTION TURN" if i == n - 1 else ""
        print(f"\n--- turn {i + 1}/{n}: {text}{marker}")
        before = set(ing.graph.edges)
        mem.ingest_session([text], f"2026-06-{i + 1:02d}")
        after_ids = set(ing.graph.edges) - before
        for eid in after_ids:
            e = ing.graph.edges[eid]
            s = ing.graph.status(eid)
            just = (" <= " + " | ".join(",".join(sorted(a))
                                        for a in e.justification)
                   if e.justification else "")
            print(f"    +{eid} ({e.subject}, {e.relation}, {e.object}) "
                  f"[{e.source_type}] sigma=({s.valid.name},{s.trans.name}){just}")
        for dec in detector.log[-5:]:
            if dec.conflict:
                print(f"    conflict: {dec.new_edge_id} vs {dec.old_edge_id} "
                      f"axis={dec.axis} via={dec.via}")

    print("\n--- final graph state ---")
    for eid, e in ing.graph.edges.items():
        s = ing.graph.status(eid)
        flag = "ACTIVE" if ing.graph.is_active(eid) else "superseded"
        print(f"  {eid} ({e.subject}, {e.relation}, {e.object}) "
              f"[{e.source_type}] sigma=({s.valid.name},{s.trans.name}) {flag}")

    print("\n--- retrieval for the question ---")
    for e in mem.retriever.retrieve(item.question,
                                    reference_time=f"2026-06-{n + 1:02d}",
                                    k=8):
        print(f"  ({e.subject} | {e.relation} | {e.object})")

    print("\n--- final answer ---")
    print(mem.answer(item.question, reference_time=f"2026-06-{n + 1:02d}"))


if __name__ == "__main__":
    main()
