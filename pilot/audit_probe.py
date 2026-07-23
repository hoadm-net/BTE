"""Audit all 168 probe items' INTERNAL graph state, not just end-to-end
QA correctness, in the same pass as the official rerun (extraction
cache is invalidated by this session's prompt/schema fixes, so a
separate audit-only pass would pay for extraction twice for no reason).

Checks two invariants that previously broke silently and only showed up
as wrong end-to-end answers several hops downstream:
  - every edge's confidence stays in [0, 1] (extraction occasionally
    emitted -1 for facts about third parties; ingest.py now clamps at
    the boundary, this re-checks the clamp actually holds under the
    real pipeline, not just in isolation)
  - no BBP propagation ever reports a `cut` (theta=0.0 is documented as
    unbounded; a cut event under theta=0.0 means propagation stopped
    for a reason other than reaching the frontier's end, which should
    be impossible now - a non-empty cut here means there is ANOTHER
    confidence-poisoning path still unaccounted for)

Usage: uv run python pilot/audit_probe.py [--limit N] [--workers N]
Writes .plan/results/runs/probe_audit.jsonl (one row per item, QA
correctness + anomaly list) and prints a summary.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

from bte.conflict import ConflictDetector, make_llm_adjudicator
from bte.extraction import extract_facts
from bte.ingest import Ingestor
from bte.llm import CachedLLM
from bte.memory import BJGMemory
from bte.probe import DOMAINS, ProbeItem, domain_rules, generate
from bte.probe_bench import ProbeJudge
from bte.retrieval import CachedEmbedder, Retriever

DOMAIN_BY_NAME = {d.name: d for d in DOMAINS}
OUT = ".plan/results/runs/probe_audit.jsonl"


@dataclass
class AuditRecord:
    probe_id: str
    hop_depth: int
    domain: str
    contradicted: str
    axis: str
    pre_correct: bool = False
    post_correct: bool = False
    pre_answer: str = ""
    post_answer: str = ""
    n_edges: int = 0
    n_derived: int = 0
    anomalies: list[str] = field(default_factory=list)
    error: str = ""


def audit_one(item: ProbeItem, extractor_kwargs: dict, reader: CachedLLM,
              judge: ProbeJudge) -> AuditRecord:
    rec = AuditRecord(probe_id=item.probe_id, hop_depth=item.hop_depth,
                      domain=item.domain, contradicted=item.contradicted,
                      axis=item.axis)
    d = DOMAIN_BY_NAME[item.domain]
    extractor = CachedLLM(**extractor_kwargs)
    vocab = list(d.relations) + list(d.conclusions)
    ing = Ingestor(
        extract=lambda text, ts, ctx: extract_facts(
            extractor, text, ts, ctx, relation_vocab=vocab),
        detector=ConflictDetector(adjudicate=make_llm_adjudicator(extractor)),
        rules=domain_rules(d),
    )
    mem = BJGMemory(ing, Retriever(ing.graph, CachedEmbedder()),
                    reader=reader.complete_text)
    n = len(item.sessions)
    try:
        for i in range(n - 1):
            text = " ".join(item.sessions[i])
            report = ing.ingest_text(text, f"2026-06-{i + 1:02d}")
            mem.retriever.index()
            for p in report.propagations:
                if p.cut:
                    rec.anomalies.append(f"bbp_cut turn{i + 1}: {p.cut}")
        rec.pre_answer = mem.answer(item.question,
                                    reference_time=f"2026-06-{n:02d}")
        rec.pre_correct = judge(item.question, item.gold_pre, rec.pre_answer)

        text = " ".join(item.sessions[n - 1])
        report = ing.ingest_text(text, f"2026-06-{n:02d}")
        mem.retriever.index()
        for p in report.propagations:
            if p.cut:
                rec.anomalies.append(f"bbp_cut turn{n}: {p.cut}")

        rec.post_answer = mem.answer(item.question,
                                     reference_time=f"2026-06-{n + 1:02d}")
        rec.post_correct = judge(item.question, item.gold_post,
                                 rec.post_answer)
    except Exception as exc:
        rec.error = f"{type(exc).__name__}: {exc}"

    rec.n_edges = len(ing.graph.edges)
    rec.n_derived = sum(1 for _ in ing.graph.derived_ids())
    bad_conf = [(e.id, e.confidence) for e in ing.graph.edges.values()
               if not (0.0 <= e.confidence <= 1.0)]
    if bad_conf:
        rec.anomalies.append(f"out_of_range_confidence: {bad_conf}")
    # a hop-N item should produce N-1 derived conclusion edges (one per
    # ChainRule) at minimum, absent extraction dropping a link entirely
    if item.hop_depth >= 2 and rec.n_derived < item.hop_depth - 1:
        rec.anomalies.append(
            f"fewer derived edges than hop depth expects: "
            f"n_derived={rec.n_derived} hop_depth={item.hop_depth}")
    return rec


def already_done(out_path: str) -> set[str]:
    if not os.path.exists(out_path):
        return set()
    latest: dict[str, dict] = {}
    with open(out_path) as f:
        for line in f:
            r = json.loads(line)
            latest[r["probe_id"]] = r
    return {pid for pid, r in latest.items() if not r.get("error")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    items = generate()
    if args.limit:
        items = items[:args.limit]
    if not args.no_resume:
        done = already_done(OUT)
        remaining = [i for i in items if i.probe_id not in done]
        print(f"resume: {len(items) - len(remaining)} already done, "
              f"{len(remaining)} remaining")
        items = remaining
        if not items:
            print("nothing left to run")
            return

    extractor_kwargs = dict(model="deepseek/deepseek-v3.2",
                            base_url="https://openrouter.ai/api/v1",
                            api_key_env="OPENROUTER_API_KEY",
                            extra={"max_tokens": 4000})
    reader = CachedLLM(model="qwen/qwen3.5-9b",
                       base_url="https://openrouter.ai/api/v1",
                       api_key_env="OPENROUTER_API_KEY")
    judge = ProbeJudge(CachedLLM(model="gpt-5", temperature=None))

    os.makedirs(".plan/results/runs", exist_ok=True)

    def run_and_write(item):
        r = audit_one(item, extractor_kwargs, reader, judge)
        with open(OUT, "a") as f:
            f.write(json.dumps(asdict(r)) + "\n")
        flag = "OK  " if r.post_correct else "MISS"
        anomaly_flag = f" ANOMALY({len(r.anomalies)})" if r.anomalies else ""
        print(f"{flag} {r.probe_id} depth={r.hop_depth}{anomaly_flag}"
              f"{' ERR ' + r.error if r.error else ''}", flush=True)
        return r

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(run_and_write, items))

    n = len(records)
    with_anomalies = [r for r in records if r.anomalies]
    print(f"\n=== summary (this run's {n} items) ===")
    print(f"pre_correct: {sum(r.pre_correct for r in records)}/{n}")
    print(f"post_correct: {sum(r.post_correct for r in records)}/{n}")
    print(f"errors: {sum(1 for r in records if r.error)}/{n}")
    print(f"items with anomalies: {len(with_anomalies)}/{n}")
    for r in with_anomalies:
        print(f"  {r.probe_id}: {r.anomalies}")


if __name__ == "__main__":
    main()
