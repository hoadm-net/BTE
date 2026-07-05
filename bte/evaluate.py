"""Reference fixed-point oracle (formalism.md, Theorem 0).

Two independent implementations of sigma*: rank-based topological
evaluation and synchronous Kleene iteration from all-BOTTOM. They must
agree (Theorem 0 uniqueness); the test suite checks that, and BBP is
tested against them. Deliberately naive — this is the specification, not
the incremental algorithm.
"""

from __future__ import annotations

from .graph import BJG, DERIVED
from .lattice import BOTTOM, Sigma, evaluate


def _inputs(g: BJG) -> dict[str, Sigma]:
    return {
        eid: g.status(eid)
        for eid, edge in g.edges.items()
        if edge.source_type != DERIVED
    }


def _phi_once(g: BJG, sigma: dict[str, Sigma]) -> dict[str, Sigma]:
    out = dict(sigma)
    for eid in g.derived_ids():
        edge = g.edges[eid]
        out[eid] = evaluate(
            [sigma[m] for m in alt] for alt in edge.justification
        )
    return out


def fixed_point_iterative(g: BJG) -> dict[str, Sigma]:
    sigma = _inputs(g)
    for eid in g.derived_ids():
        sigma[eid] = BOTTOM
    while True:
        nxt = _phi_once(g, sigma)
        if nxt == sigma:
            return sigma
        sigma = nxt


def fixed_point_rank(g: BJG) -> dict[str, Sigma]:
    sigma = _inputs(g)
    pending = set(g.derived_ids())
    # insertion order of a BJG is already topological (members must exist
    # before their dependents), so one ordered pass settles every edge
    for eid in g.edges:
        if eid in pending:
            edge = g.edges[eid]
            sigma[eid] = evaluate(
                [sigma[m] for m in alt] for alt in edge.justification
            )
    return sigma
