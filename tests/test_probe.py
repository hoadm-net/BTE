"""Generator validation: every probe item's gold labels must be
reproduced by replaying its oracle facts through the Ingestor with the
domain's rule set — the generator is checked against the proven core,
not against its own bookkeeping.
"""

import pytest

from bte.ingest import Ingestor
from bte.probe import DOMAINS, domain_rules, generate

DOMAIN_BY_NAME = {d.name: d for d in DOMAINS}
ITEMS = generate()


def replay(item):
    ing = Ingestor(rules=domain_rules(DOMAIN_BY_NAME[item.domain]))
    n = len(item.oracle_facts)
    for i in range(n - 1):
        report = ing.ingest_facts(item.oracle_facts[i], f"t{i}")
        ing.apply_retractions(item.oracle_retractions[i], report)
    pre = active_values(ing, item)
    report = ing.ingest_facts(item.oracle_facts[n - 1], f"t{n - 1}")
    ing.apply_retractions(item.oracle_retractions[n - 1], report)
    return ing, pre


def probed_relation(item):
    d = DOMAIN_BY_NAME[item.domain]
    return d.relations[0] if item.hop_depth == 1 \
        else d.conclusions[item.hop_depth - 2]


def active_values(ing, item):
    return [e.object for e in
            ing.graph.find(subject="user", relation=probed_relation(item))]


def test_factorial_coverage():
    assert len(ITEMS) == 168
    margins = {}
    for it in ITEMS:
        margins.setdefault((it.hop_depth, it.axis), []).append(it)
    assert len(margins) == 8
    for cell, members in margins.items():
        floor = 12 if cell[0] == 1 else 24
        assert len(members) >= floor, cell


def test_determinism():
    a = [i.__dict__ for i in generate(seed=1)]
    b = [i.__dict__ for i in generate(seed=1)]
    assert a == b


@pytest.mark.parametrize("item", ITEMS, ids=lambda i: i.probe_id)
def test_gold_labels_replay(item):
    ing, pre = replay(item)

    # pre-contradiction: the probed conclusion holds with the gold value
    assert pre == [item.gold_pre], f"pre state {pre}"

    post = active_values(ing, item)
    if item.gold_post == "unknown":
        assert post == [], f"post state {post}"
    else:
        assert post == [item.gold_post], f"post state {post}"

    # every relation the gold marks invalidated lost its old active edge
    for rel in item.gold_invalidated:
        stale = [
            e for e in ing.graph.find(relation=rel)
            if e.object in (item.gold_pre,) and item.gold_post != e.object
        ]
        assert not stale, f"{rel} still active with pre value"
