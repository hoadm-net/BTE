"""Bounded Bitemporal Propagation (proposed-model.md section 3;
formalism.md Theorems 2-3).

Triggered after a contradiction event forces an edge's status down.
Unbounded (max_depth=None, theta=0) this is chaotic iteration of the
evaluation operator over the affected component and must reproduce the
oracle exactly (Theorem 2) — the property-based tests enforce that.
Bounded, the un-reevaluated dependents form the cut set K; edges outside
K and its descendants are still exact (Theorem 3's stale-by-omission
proposition), which the tests also enforce.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from .graph import BJG
from .lattice import S, Sigma, leq


@dataclass
class BBPResult:
    changed: dict[str, tuple[Sigma, Sigma]] = field(default_factory=dict)
    visits: int = 0
    waves: int = 0
    cut: set[str] = field(default_factory=set)


def bbp(
    g: BJG,
    event_edge: str,
    new_status: Sigma,
    max_depth: Optional[int] = None,
    theta: float = 0.0,
    decay_factor: float = 0.9,
) -> BBPResult:
    old = g.status(event_edge)
    if not leq(new_status, old):
        raise ValueError("contradiction events may only lower status")

    result = BBPResult()
    if new_status == old:
        return result
    g.force_status(event_edge, new_status)
    result.changed[event_edge] = (old, new_status)

    frontier: deque[str] = deque(sorted(g.dependents(event_edge)))
    depth = 1
    while frontier and (max_depth is None or depth <= max_depth):
        result.waves += 1
        next_frontier: set[str] = set()
        for d in frontier:
            result.visits += 1
            before = g.status(d)
            after = g._evaluate(d)
            if after == before:
                continue
            g.force_status(d, after)
            prior = result.changed.get(d)
            result.changed[d] = (prior[0] if prior else before, after)
            edge = g.edges[d]
            if before.trans != S.BOT and after.trans == S.BOT:
                edge.confidence *= decay_factor
            if edge.confidence < theta:
                result.cut |= g.dependents(d)
                continue
            next_frontier |= g.dependents(d)
        frontier = deque(sorted(next_frontier))
        depth += 1

    if frontier and max_depth is not None:
        result.cut |= set(frontier)
    return result
