"""Closed-set derivation rules (proposed-model.md 2.2: temporal
composition / attribute inheritance are instances of a two-premise chain).

A ChainRule fires when active edges (s, r1, x) and (x, r2, o) exist,
concluding (s, r_out, o) justified by exactly those two edges. Repeated
closure passes handle chains of derived edges. A conclusion reachable via
several premise pairs gets each pair as an independent justification
alternative — required for alternative-support checking in BBP.
"""

from __future__ import annotations

from dataclasses import dataclass

from .extraction import normalize
from .graph import BJG, DERIVED, Edge


@dataclass(frozen=True)
class ChainRule:
    r1: str
    r2: str
    r_out: str


def _later(a: str | None, b: str | None) -> str | None:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def _earlier(a: str | None, b: str | None) -> str | None:
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def derive_closure(g: BJG, rules: list[ChainRule],
                   next_id, t_transaction: str | None = None) -> list[str]:
    """Apply rules to a fixpoint. `next_id` supplies fresh edge ids.
    Returns ids of edges added this call.
    """
    added: list[str] = []
    changed = True
    while changed:
        changed = False
        for rule in rules:
            for left in g.find(relation=rule.r1):
                mid = normalize(left.object)
                for right in g.find(subject=mid, relation=rule.r2):
                    alt = frozenset({left.id, right.id})
                    existing = [
                        e for e in g.find(subject=left.subject,
                                          relation=rule.r_out,
                                          active_only=False)
                        if e.object == right.object
                        and e.source_type == DERIVED
                    ]
                    if existing:
                        if alt not in existing[0].justification:
                            g.add_alternative(existing[0].id, alt)
                            changed = True
                        continue
                    eid = next_id()
                    g.add_derived(Edge(
                        id=eid,
                        subject=left.subject,
                        relation=rule.r_out,
                        object=right.object,
                        # inherited window: intersection of premise windows
                        t_valid_start=_later(left.t_valid_start,
                                             right.t_valid_start),
                        t_valid_end=_earlier(left.t_valid_end,
                                             right.t_valid_end),
                        t_transaction=t_transaction,
                        confidence=min(left.confidence, right.confidence),
                        justification=(alt,),
                    ))
                    added.append(eid)
                    changed = True
    return added
