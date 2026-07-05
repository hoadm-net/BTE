"""Executable versions of formalism.md's counterexample (Theorem 1) and
the alt-support behavior that separates the propagation rule from naive
delete-everything cascading.
"""

import pytest

from bte import BJG, TOP, Edge, S, Sigma, bbp, fixed_point_rank


@pytest.mark.parametrize("axis", ["valid", "trans"])
def test_theorem1_counterexample(axis):
    g = BJG()
    g.add_asserted(Edge(id="A"), TOP)
    g.add_derived(Edge(id="B", justification=(frozenset({"A"}),)))
    assert g.status("B") == TOP

    lowered = Sigma(S.BOT, S.TOP) if axis == "valid" else Sigma(S.TOP, S.BOT)

    # single-edge resolution: touch A, leave B — B no longer matches sigma*
    g.force_status("A", lowered)
    stale = g.status("B")
    oracle = fixed_point_rank(g)
    assert stale == TOP
    assert oracle["B"] == lowered
    assert stale != oracle["B"]

    # propagation restores the fixed point
    g.force_status("A", TOP)
    g.force_status("B", TOP)
    bbp(g, "A", lowered)
    assert g.status("B") == lowered


def test_alt_support_survives():
    """An edge entailed two independent ways survives one premise falling."""
    g = BJG()
    g.add_asserted(Edge(id="A"), TOP)
    g.add_asserted(Edge(id="C"), TOP)
    g.add_derived(
        Edge(id="B", justification=(frozenset({"A"}), frozenset({"C"})))
    )
    result = bbp(g, "A", Sigma(S.TOP, S.BOT))
    assert g.status("B") == TOP
    assert "B" not in result.changed
    # and the cascade stopped: nothing beyond B was even visited
    assert result.visits == 1


def test_two_hop_chain_propagates():
    g = BJG()
    g.add_asserted(Edge(id="marathon"), TOP)
    g.add_derived(Edge(id="longruns", justification=(frozenset({"marathon"}),)))
    g.add_derived(Edge(id="no-sat-meetings", justification=(frozenset({"longruns"}),)))
    lowered = Sigma(S.TOP, S.BOT)
    bbp(g, "marathon", lowered)
    assert g.status("longruns") == lowered
    assert g.status("no-sat-meetings") == lowered


def test_confidence_decay_on_supersession():
    g = BJG()
    g.add_asserted(Edge(id="A"), TOP)
    g.add_derived(Edge(id="B", justification=(frozenset({"A"}),)))
    assert g.edges["B"].confidence == 1.0
    bbp(g, "A", Sigma(S.TOP, S.BOT), decay_factor=0.5)
    assert g.edges["B"].confidence == 0.5
