"""Run a MemorySystem over a LongMemEval-S slice on the shared harness.

Usage:
  uv run python pilot/run_longmemeval.py --limit 5 --category knowledge-update
Appends records to .plan/results/runs/longmemeval_s.jsonl and prints the
summary. Systems: bjg (ours). Baseline adapters land in Phase 3 proper.
"""

import argparse
import json
import os
import random

from bte.bench import Judge, load_longmemeval, run_benchmark, summarize
from bte.conflict import ConflictDetector, make_llm_adjudicator
from bte.extraction import extract_facts
from bte.ingest import Ingestor
from bte.llm import CachedLLM
from bte.memory import BJGMemory
from bte.retrieval import CachedEmbedder, Retriever

DATA = "data/longmemeval/longmemeval_s_cleaned.json"
OUT = ".plan/results/runs/longmemeval_s.jsonl"


def bjg_factory(extractor: CachedLLM, reader: CachedLLM):
    def make():
        ing = Ingestor(
            extract=lambda text, ts, ctx: extract_facts(
                extractor, text, ts, ctx),
            detector=ConflictDetector(
                adjudicate=make_llm_adjudicator(extractor)),
        )
        return BJGMemory(ing, Retriever(ing.graph, CachedEmbedder()),
                         reader=reader.complete_text)
    return make


def mem0_factory(reader: CachedLLM):
    from bte.baselines.mem0_adapter import Mem0Memory

    def make():
        return Mem0Memory(reader=reader.complete_text)
    return make


def already_done(out_path: str, system: str,
                 max_ingest_errors: int = 3) -> set[str]:
    """question_ids whose MOST RECENT record for this system is a
    genuinely usable success. Baselines with no call-level cache of
    their own (Mem0's internal extraction is not routed through
    CachedLLM) redo real, billed work on every rerun otherwise - a crash
    partway through a 100-question run should not mean re-paying for the
    ones that already succeeded.

    Must be latest-wins, not "ever succeeded": a question_id can have an
    old successful record from before a code/prompt change and a newer
    failing one from after it. "Ever succeeded" resume would treat it as
    done and never retry the failing latest attempt, silently keeping
    stale results (observed with the probe harness: an enum-constrained
    extraction change made 33 previously-clean items fail, and naive
    resume kept serving their pre-change successes forever).

    error=None alone is also not enough: a session-level failure inside
    ingest_session is caught per-session (bench.py), so a question can
    finish with no top-level error while most of its sessions were
    silently dropped (observed: an OpenRouter outage mid-run left ~30
    'successful' mem0 records with 40+ of ~48 sessions skipped, all
    answering 'unknown' from a near-empty memory) - those are not done,
    they are quietly-corrupted and must be redone.
    """
    if not os.path.exists(out_path):
        return set()
    latest: dict[str, dict] = {}
    with open(out_path) as f:
        for line in f:
            r = json.loads(line)
            if r["system"] == system:
                latest[r["question_id"]] = r
    return {qid for qid, r in latest.items()
            if not r.get("error")
            and r.get("ingest_errors", 0) <= max_ingest_errors}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="bjg", choices=["bjg", "mem0"])
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--category", default=None)
    ap.add_argument("--seed", type=int, default=20260705)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-resume", action="store_true",
                    help="reprocess every selected question even if a "
                         "successful record already exists")
    args = ap.parse_args()

    questions = load_longmemeval(DATA)
    rng = random.Random(args.seed)
    if args.category:
        cats = args.category.split(",")
        per_cat = max(1, args.limit // len(cats))
        picked = []
        for c in cats:
            pool = [q for q in questions if q["question_type"] == c]
            rng.shuffle(pool)
            picked += pool[:per_cat]
        questions = picked
    else:
        rng.shuffle(questions)
        questions = questions[:args.limit]

    if not args.no_resume:
        done = already_done(OUT, args.system)
        remaining = [q for q in questions
                     if str(q["question_id"]) not in done]
        print(f"resume: {len(done & {str(q['question_id']) for q in questions})} "
              f"already done, {len(remaining)} remaining")
        questions = remaining
        if not questions:
            print("nothing left to run")
            return

    extractor = CachedLLM(model="deepseek/deepseek-v3.2",
                          base_url="https://openrouter.ai/api/v1",
                          api_key_env="OPENROUTER_API_KEY",
                          extra={"max_tokens": 4000})
    reader = CachedLLM(model="qwen/qwen3.5-9b",
                       base_url="https://openrouter.ai/api/v1",
                       api_key_env="OPENROUTER_API_KEY")
    judge = Judge(CachedLLM(model="gpt-5", temperature=None))

    factory = (mem0_factory(reader) if args.system == "mem0"
               else bjg_factory(extractor, reader))

    import os
    os.makedirs(".plan/results/runs", exist_ok=True)
    records = run_benchmark(
        factory, questions, judge,
        system=args.system, out_path=OUT, max_workers=args.workers)
    print(json.dumps(summarize(records), indent=1))
    for r in records:
        flag = "OK " if r.correct else "MISS"
        print(f"{flag} [{r.category}] {r.question_id} "
              f"gold={r.gold!r} got={r.model_answer[:80]!r} "
              f"ingest={r.ingest_seconds}s{' ERR ' + r.error if r.error else ''}")


if __name__ == "__main__":
    main()
