"""Stratified sample selection for probe human validation.

Selects >=30% of the 168 probe items, proportionally from every
(domain, hop_depth) cell rather than the first N in file order - a run
of items from one axis combination would leave other combinations
unchecked. Deterministic (seeded), so re-running produces the same
sample rather than a new one each time the app restarts.
"""

from __future__ import annotations

import random
from collections import defaultdict

SEED = 20260705
MIN_FRACTION = 0.30


def stratified_sample(items: list[dict], seed: int = SEED,
                      fraction: float = MIN_FRACTION) -> list[dict]:
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for item in items:
        cells[(item["domain"], item["hop_depth"])].append(item)

    rng = random.Random(seed)
    picked: list[dict] = []
    for key in sorted(cells):
        cell = cells[key]
        rng.shuffle(cell)
        n = max(1, round(len(cell) * fraction))
        picked.extend(cell[:n])

    # top up from the largest under-sampled cells if rounding left the
    # overall sample just short of the target fraction
    target = round(len(items) * fraction)
    if len(picked) < target:
        picked_ids = {i["probe_id"] for i in picked}
        leftovers = [i for i in items if i["probe_id"] not in picked_ids]
        rng.shuffle(leftovers)
        picked.extend(leftovers[:target - len(picked)])

    picked.sort(key=lambda i: i["probe_id"])
    return picked
