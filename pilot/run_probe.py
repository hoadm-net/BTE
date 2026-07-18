"""Run a MemorySystem over the diagnostic probe (168 items, hop depth
1-4) on the shared harness. This is the H1/H2 instrument: unlike
LongMemEval's KU/TR, hop depth is a controlled design axis here.

Usage:
  uv run python pilot/run_probe.py --system bjg --limit 20
  uv run python pilot/run_probe.py --system mem0 --limit 20
Appends records to .plan/results/runs/probe_v0.jsonl and prints the
summary (accuracy overall + by hop depth).
"""

import argparse
import json
import os

from bte.conflict import ConflictDetector, make_llm_adjudicator
from bte.extraction import extract_facts
from bte.ingest import Ingestor
from bte.llm import CachedLLM
from bte.memory import BJGMemory
from bte.probe import DOMAINS, domain_rules, generate
from bte.probe_bench import ProbeJudge, run_probe_benchmark, summarize_probe
from bte.retrieval import CachedEmbedder, Retriever

OUT = ".plan/results/runs/probe_v0.jsonl"
DOMAIN_BY_NAME = {d.name: d for d in DOMAINS}


def bjg_factory(extractor_kwargs: dict, reader: CachedLLM):
    def make(item):
        d = DOMAIN_BY_NAME[item.domain]
        # Fresh extractor client per item, not one shared across the
        # whole run: a long-lived client's pooled HTTP connection
        # appears to go bad after enough requests (observed: the same
        # ~30 items failed identically, deterministically, on every
        # rerun through a shared client - including fully sequential
        # reruns with no concurrency at all - yet every one of them
        # succeeds immediately in isolation with a fresh client). The
        # disk cache is keyed on request content, not on the client
        # instance, so this costs nothing on cache hits.
        extractor = CachedLLM(**extractor_kwargs)
        # The probe's domain schema is known in advance, so relation
        # names can be structurally constrained (extraction.py's
        # relation_vocab / EDC-style target alignment) rather than
        # relying on the model to freely reproduce the exact string —
        # this is what makes derive_closure()'s exact-match ChainRule
        # lookups reliable. Open-domain data (LongMemEval etc.) has no
        # such fixed vocabulary and doesn't get this treatment.
        vocab = list(d.relations) + list(d.conclusions)
        ing = Ingestor(
            extract=lambda text, ts, ctx: extract_facts(
                extractor, text, ts, ctx, relation_vocab=vocab),
            detector=ConflictDetector(
                adjudicate=make_llm_adjudicator(extractor)),
            rules=domain_rules(d),
        )
        return BJGMemory(ing, Retriever(ing.graph, CachedEmbedder()),
                         reader=reader.complete_text)
    return make


def mem0_factory(reader: CachedLLM):
    from bte.baselines.mem0_adapter import Mem0Memory

    def make(item):
        return Mem0Memory(reader=reader.complete_text)
    return make


def already_done(out_path: str, system: str) -> set[str]:
    """probe_ids whose MOST RECENT record for this system succeeded.
    Must be latest-wins, not "ever succeeded": a probe_id can have an
    old successful record from before a code/prompt change and a newer
    failing one from after it (observed: an enum-constrained extraction
    run failed on 33 items that had succeeded in earlier runs before the
    fix landed - "ever succeeded" resume treated them as done and never
    retried the failing latest attempt, silently keeping stale results)."""
    if not os.path.exists(out_path):
        return set()
    latest: dict[str, dict] = {}
    with open(out_path) as f:
        for line in f:
            r = json.loads(line)
            if r["system"] == system:
                latest[r["probe_id"]] = r
    return {pid for pid, r in latest.items() if not r.get("error")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="bjg", choices=["bjg", "mem0"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--depths", default=None,
                    help="comma-separated hop depths to include, e.g. 2,3,4")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    items = generate()
    if args.depths:
        wanted = {int(x) for x in args.depths.split(",")}
        items = [i for i in items if i.hop_depth in wanted]
    if args.limit:
        items = items[:args.limit]

    if not args.no_resume:
        done = already_done(OUT, args.system)
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

    factory = mem0_factory(reader) if args.system == "mem0" \
        else bjg_factory(extractor_kwargs, reader)

    os.makedirs(".plan/results/runs", exist_ok=True)
    records = run_probe_benchmark(
        factory, items, judge, system=args.system, out_path=OUT,
        max_workers=args.workers)

    print(json.dumps(summarize_probe(records), indent=1))
    for r in records:
        flag = "OK  " if r.post_correct else "MISS"
        print(f"{flag} {r.probe_id} depth={r.hop_depth} axis={r.axis} "
              f"gold_post={r.gold_post!r} got={r.post_answer[:60]!r}"
              f"{' ERR ' + r.error if r.error else ''}")


if __name__ == "__main__":
    main()
