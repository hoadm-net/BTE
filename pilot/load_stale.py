"""Loader + stats for STALE (T1_T2_400_FULL.json).

Resolves the flagged unknowns in docs/evaluation-benchmarks.md: actual
context-length distribution (docs assumed ~75K tokens/scenario as a
placeholder — the largest cost uncertainty), scenario/query counts, and
probing-dimension breakdown.

Usage: python3 pilot/load_stale.py data/stale/T1_T2_400_FULL.json
"""

import json
import sys
from collections import Counter


def approx_tokens(text):
    return len(text) // 4


def walk_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from walk_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk_strings(v)


def main(path):
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, dict):
        print(f"top-level: dict with keys {list(data.keys())[:10]}")
        items = next(v for v in data.values() if isinstance(v, list))
    else:
        items = data
    print(f"file: {path}")
    print(f"items: {len(items)}")
    print(f"item fields: {sorted(items[0].keys())}")

    # field value distributions for small string fields (find the taxonomy)
    field_vals = {}
    for k in items[0]:
        vals = [i.get(k) for i in items[:400]]
        if all(isinstance(v, (str, int, type(None))) for v in vals):
            c = Counter(vals)
            if 1 < len(c) <= 15:
                field_vals[k] = c
    for k, c in field_vals.items():
        print(f"\ndistribution of '{k}':")
        for v, n in c.most_common():
            print(f"  {v}: {n}")

    # context size per item (all strings, chars/4)
    sizes = sorted(
        sum(approx_tokens(s) for s in walk_strings(i)) for i in items)
    n = len(sizes)
    print(f"\napprox tokens/item: min {sizes[0] / 1e3:.0f}K, "
          f"p25 {sizes[n // 4] / 1e3:.0f}K, median {sizes[n // 2] / 1e3:.0f}K, "
          f"p75 {sizes[3 * n // 4] / 1e3:.0f}K, max {sizes[-1] / 1e3:.0f}K, "
          f"sum {sum(sizes) / 1e6:.1f}M")


if __name__ == "__main__":
    main(sys.argv[1])
