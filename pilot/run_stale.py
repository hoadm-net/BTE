"""Run a MemorySystem over STALE T2 scenarios (external, expert-validated
multi-hop evidence - see .plan's theory-fit audit: 188/200 T2 items carry
an explicit dependency annotation between the old and new state).

Usage:
  uv run python pilot/run_stale.py --system bjg --limit 10
  uv run python pilot/run_stale.py --system mem0 --limit 10
Appends records to .plan/results/runs/stale_v0.jsonl.
"""

import argparse
import json
import os

from bte.conflict import (ConflictDetector, DomainDependencies,
                          make_llm_adjudicator, make_llm_proposer)
from bte.extraction import extract_facts
from bte.ingest import Ingestor
from bte.llm import CachedLLM
from bte.memory import BJGMemory
from bte.retrieval import CachedEmbedder, Retriever
from bte.stale_bench import (STALE_READER_SYSTEM, StaleJudge,
                             run_stale_benchmark, summarize_stale)

DATA = "data/stale/T1_T2_400_FULL.json"
OUT = ".plan/results/runs/stale_v0.jsonl"


def bjg_factory(extractor: CachedLLM, reader: CachedLLM,
                semantic: bool = False, domain: bool = True):
    def make(scenario):
        # No fixed relation vocabulary for open-domain conversational
        # data (unlike the probe) - relation_vocab / enum-constrained
        # extraction only applies where the target vocabulary is known
        # in advance. The coarse life-domain enum is a different layer:
        # it types every fact regardless of relation surface form, and
        # gates the implicit-conflict candidate search (see
        # conflict.py's module docstring for why the domain path exists
        # and what it fixes over embedding-similarity candidates).
        #
        # embedder is shared between retrieval and (when enabled) the
        # semantic conflict path: same on-disk cache, so a fact's
        # embedding computed for one purpose is free for the other.
        embedder = CachedEmbedder()
        ing = Ingestor(
            extract=lambda text, ts, ctx: extract_facts(
                extractor, text, ts, ctx),
            detector=ConflictDetector(
                adjudicate=make_llm_adjudicator(extractor),
                embedder=embedder if semantic else None,
                domain_deps=DomainDependencies() if domain else None,
                propose=make_llm_proposer(extractor) if domain else None),
        )
        return BJGMemory(ing, Retriever(ing.graph, embedder),
                         reader=reader.complete_text,
                         reader_system=STALE_READER_SYSTEM)
    return make


def mem0_factory(reader: CachedLLM):
    from bte.baselines.mem0_adapter import Mem0Memory

    def make(scenario):
        return Mem0Memory(reader=reader.complete_text,
                          reader_system=STALE_READER_SYSTEM)
    return make


def already_done(out_path: str, system: str) -> set[str]:
    """uids whose MOST RECENT record for this system succeeded - see
    run_probe.py's version for why this must be latest-wins."""
    if not os.path.exists(out_path):
        return set()
    latest: dict[str, dict] = {}
    with open(out_path) as f:
        for line in f:
            r = json.loads(line)
            if r["system"] == system:
                latest[r["uid"]] = r
    return {uid for uid, r in latest.items() if not r.get("error")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="bjg", choices=["bjg", "mem0"])
    ap.add_argument("--stale-type", default="T2", choices=["T1", "T2"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260705)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-resume", action="store_true")
    # conflict-channel ablation toggles (bjg only). The system label
    # records the configuration so records for different configurations
    # never collide in the results file or in resume logic.
    ap.add_argument("--semantic", action="store_true",
                    help="enable the embedding-similarity conflict path")
    ap.add_argument("--no-domain", action="store_true",
                    help="disable the domain-typed conflict path")
    args = ap.parse_args()

    system = args.system
    if system == "bjg":
        if not args.no_domain:
            system += "-dom"
        if args.semantic:
            system += "-sem"

    with open(DATA) as f:
        data = json.load(f)
    scenarios = [s for s in data if s["type"] == args.stale_type]

    import random
    random.Random(args.seed).shuffle(scenarios)
    if args.limit:
        scenarios = scenarios[:args.limit]

    if not args.no_resume:
        done = already_done(OUT, system)
        remaining = [s for s in scenarios if s["uid"] not in done]
        print(f"resume: {len(scenarios) - len(remaining)} already done, "
              f"{len(remaining)} remaining")
        scenarios = remaining
        if not scenarios:
            print("nothing left to run")
            return

    extractor = CachedLLM(model="deepseek/deepseek-v3.2",
                          base_url="https://openrouter.ai/api/v1",
                          api_key_env="OPENROUTER_API_KEY",
                          extra={"max_tokens": 4000})
    reader = CachedLLM(model="qwen/qwen3.5-9b",
                       base_url="https://openrouter.ai/api/v1",
                       api_key_env="OPENROUTER_API_KEY")
    judge = StaleJudge(CachedLLM(model="gpt-5", temperature=None))

    factory = mem0_factory(reader) if args.system == "mem0" \
        else bjg_factory(extractor, reader, semantic=args.semantic,
                         domain=not args.no_domain)

    os.makedirs(".plan/results/runs", exist_ok=True)
    records = run_stale_benchmark(
        factory, scenarios, judge, system=system, out_path=OUT,
        max_workers=args.workers)

    print(json.dumps(summarize_stale(records), indent=1))
    for r in records:
        d1 = "OK" if r.dim1_correct else "MISS"
        d2 = "OK" if r.dim2_correct else "MISS"
        d3 = "OK" if r.dim3_correct else "MISS"
        print(f"{r.uid[:8]} d1={d1} d2={d2} d3={d3} "
              f"ingest={r.ingest_seconds}s{' ERR ' + r.error if r.error else ''}")


if __name__ == "__main__":
    main()
