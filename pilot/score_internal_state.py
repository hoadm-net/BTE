"""Internal-state scoring for STALE T2: did ingestion actually retire
M_old? Rebuilds each scenario's graph from the LLM cache (near-free) and
scores the memory layer directly against the scenario's gold fields,
independent of reader and judge noise - the layer-attribution number
the end-to-end dims cannot give.

M_old has no gold edge id, so it is operationalized as the top-2
old-session edges by embedding similarity to the scenario's M_old text
(embeddings cached). Reported per scenario:
  - m_old_superseded: any representative edge no longer active
  - via: which conflict channel superseded it (from the decision log)
  - m_new_intact: the new-session edge most similar to M_new is still
    active (guards against the self-destruction regression)

Usage: uv run python pilot/score_internal_state.py [--limit 10]
Writes .plan/results/runs/internal_state_v2.json and prints a table.
"""

import argparse
import json
import os
import random

from bte.bench import render_session
from bte.canonical import (RelationCanonicalizer,
                           make_llm_relation_verifier)
from bte.conflict import (ConflictDetector, make_llm_adjudicator,
                          make_llm_proposer)
from bte.domains import DomainDependencies
from bte.extraction import extract_facts
from bte.ingest import Ingestor
from bte.llm import CachedLLM
from bte.retrieval import CachedEmbedder, cosine, edge_text

DATA = "data/stale/T1_T2_400_FULL.json"
OUT = ".plan/results/runs/internal_state_v2.json"


def build(scenario, extractor, embedder):
    deps = DomainDependencies()
    ing = Ingestor(
        extract=lambda text, ts, ctx: extract_facts(extractor, text, ts, ctx),
        detector=ConflictDetector(adjudicate=make_llm_adjudicator(extractor),
                                  domain_deps=deps,
                                  propose=make_llm_proposer(extractor)),
        canonicalizer=RelationCanonicalizer(
            embedder=embedder,
            verify=make_llm_relation_verifier(extractor)),
    )
    errors = 0
    for session, ts in zip(scenario["haystack_session"],
                           scenario["timestamps"]):
        try:
            ing.ingest_text(render_session(session), ts)
        except Exception:
            errors += 1
    return ing, errors


def representatives(graph, embedder, session_ts, state_text, top: int):
    edges = [e for e in graph.edges.values()
             if e.t_transaction == session_ts and e.subject == "user"]
    if not edges:
        return []
    vecs = embedder.embed([edge_text(e) for e in edges] + [state_text])
    ranked = sorted(zip(edges, vecs[:-1]),
                    key=lambda ev: -cosine(vecs[-1], ev[1]))
    return [e for e, _ in ranked[:top]]


def score_one(scenario, extractor, embedder):
    ing, errors = build(scenario, extractor, embedder)
    g = ing.graph
    old_idx, new_idx = scenario["relevant_session_index"][:2]
    old_ts = scenario["timestamps"][old_idx]
    new_ts = scenario["timestamps"][new_idx]

    old_reps = representatives(g, embedder, old_ts, scenario["M_old"], 2)
    new_reps = representatives(g, embedder, new_ts, scenario["M_new"], 1)

    superseded = [e for e in old_reps if not g.is_active(e.id)]
    via = None
    for d in ing.detector.log:
        if d.conflict and any(d.old_edge_id == e.id for e in superseded):
            via = d.via
            break
    return {
        "uid": scenario["uid"],
        "ingest_errors": errors,
        "m_old_reps": [f"{e.relation}={e.object[:60]}" for e in old_reps],
        "m_old_superseded": bool(superseded),
        "superseded_via": via,
        "m_new_intact": bool(new_reps) and all(
            g.is_active(e.id) for e in new_reps),
        "graph_edges": len(g.edges),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260705)
    args = ap.parse_args()

    data = json.load(open(DATA))
    scenarios = [s for s in data if s["type"] == "T2"]
    random.Random(args.seed).shuffle(scenarios)
    scenarios = scenarios[:args.limit]

    extractor = CachedLLM(model="deepseek/deepseek-v3.2",
                          base_url="https://openrouter.ai/api/v1",
                          api_key_env="OPENROUTER_API_KEY",
                          extra={"max_tokens": 4000})
    embedder = CachedEmbedder()

    results = []
    for s in scenarios:
        r = score_one(s, extractor, embedder)
        results.append(r)
        print(f"{r['uid'][:8]} m_old_superseded={r['m_old_superseded']} "
              f"via={r['superseded_via']} m_new_intact={r['m_new_intact']} "
              f"errors={r['ingest_errors']}")

    n = len(results)
    summary = {
        "n": n,
        "m_old_superseded_rate": round(
            sum(r["m_old_superseded"] for r in results) / n, 3),
        "m_new_intact_rate": round(
            sum(r["m_new_intact"] for r in results) / n, 3),
    }
    print(json.dumps(summary, indent=1))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
