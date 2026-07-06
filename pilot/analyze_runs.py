"""Read the shared results JSONL and report accuracy plus layer
attribution: for each miss, was the gold value in the graph (retrieval or
reader lost it) or not (extraction lost it)? Deduplicates by
(system, question_id), keeping the most recent record so reruns do not
double-count.

Usage: uv run python pilot/analyze_runs.py [.plan/results/runs/longmemeval_s.jsonl]
"""

import json
import sys
from collections import defaultdict

PATH = sys.argv[1] if len(sys.argv) > 1 else \
    ".plan/results/runs/longmemeval_s.jsonl"


def main():
    latest = {}
    with open(PATH) as f:
        for line in f:
            r = json.loads(line)
            latest[(r["system"], r["question_id"])] = r
    records = list(latest.values())
    if not records:
        print("no records")
        return

    by_system = defaultdict(list)
    for r in records:
        by_system[r["system"]].append(r)

    for system, rs in sorted(by_system.items()):
        n = len(rs)
        correct = sum(r["correct"] for r in rs)
        print(f"\n=== {system}  ({correct}/{n} = {correct / n:.3f}) ===")

        cats = defaultdict(lambda: [0, 0])
        for r in rs:
            cats[r["category"]][1] += 1
            cats[r["category"]][0] += r["correct"]
        for c, (ok, tot) in sorted(cats.items()):
            print(f"  {c}: {ok}/{tot}")

        misses = [r for r in rs if not r["correct"]]
        # layer attribution over misses with a usable gold_in_graph signal
        extraction = sum(1 for r in misses if r.get("gold_in_graph") is False)
        downstream = sum(1 for r in misses if r.get("gold_in_graph") is True)
        unknown = sum(1 for r in misses if r.get("gold_in_graph") is None)
        print(f"  misses: {len(misses)} | extraction-lost {extraction} "
              f"| retrieval/reader-lost {downstream} | unattributed {unknown}")
        errs = sum(1 for r in rs if r.get("error"))
        ingest_errs = sum(r.get("ingest_errors", 0) for r in rs)
        print(f"  hard errors: {errs} | skipped sessions: {ingest_errs}")


if __name__ == "__main__":
    main()
