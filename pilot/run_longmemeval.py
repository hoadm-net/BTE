"""Run a MemorySystem over a LongMemEval-S slice on the shared harness.

Usage:
  uv run python pilot/run_longmemeval.py --limit 5 --category knowledge-update
Appends records to .plan/results/runs/longmemeval_s.jsonl and prints the
summary. Systems: bjg (ours). Baseline adapters land in Phase 3 proper.
"""

import argparse
import json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="bjg", choices=["bjg", "mem0"])
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--category", default=None)
    ap.add_argument("--seed", type=int, default=20260705)
    ap.add_argument("--workers", type=int, default=4)
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
