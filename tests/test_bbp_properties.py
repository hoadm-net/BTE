"""Property-based checks of Theorems 0, 2, 3 over random justification
DAGs: the two oracle implementations agree; unbounded BBP reproduces the
oracle exactly; repeated BBP is a no-op; bounded BBP's stale edges are
confined to the cut set's cone of influence.

Example counts come from the hypothesis profiles in conftest.py; run with
HYPOTHESIS_PROFILE=thorough for the high-volume verification pass.
"""

from hypothesis import given
from hypothesis import strategies as st

from bte import ALL_SIGMA, BJG, Edge, bbp, fixed_point_iterative, fixed_point_rank, leq

sigmas = st.sampled_from(ALL_SIGMA)


@st.composite
def graph_and_event(draw):
    g = BJG()
    ids = []
    for i in range(draw(st.integers(2, 5))):
        eid = f"a{i}"
        g.add_asserted(Edge(id=eid), draw(sigmas))
        ids.append(eid)
    for j in range(draw(st.integers(1, 12))):
        alts = []
        for _ in range(draw(st.integers(1, 3))):
            alts.append(frozenset(draw(
                st.lists(st.sampled_from(ids), min_size=1, max_size=3,
                         unique=True))))
        eid = f"d{j}"
        g.add_derived(Edge(id=eid, justification=tuple(alts)))
        ids.append(eid)
    target = draw(st.sampled_from([i for i in ids if i.startswith("a")]))
    lowered = draw(st.sampled_from(
        [s for s in ALL_SIGMA if leq(s, g.status(target))]))
    return g, target, lowered


@given(graph_and_event())
def test_oracles_agree(case):
    g, _, _ = case
    assert fixed_point_rank(g) == fixed_point_iterative(g)


@given(graph_and_event())
def test_unbounded_bbp_computes_fixed_point(case):
    g, target, lowered = case
    bbp(g, target, lowered)
    oracle = fixed_point_rank(g)
    for eid in g.edges:
        assert g.status(eid) == oracle[eid], eid


@given(graph_and_event())
def test_bbp_is_idempotent(case):
    g, target, lowered = case
    bbp(g, target, lowered)
    again = bbp(g, target, lowered)
    assert not again.changed


@given(graph_and_event(), st.integers(0, 2))
def test_bounded_staleness_confined_to_cut_cone(case, max_depth):
    g, target, lowered = case
    result = bbp(g, target, lowered, max_depth=max_depth)
    oracle = fixed_point_rank(g)
    allowed = set(result.cut)
    for c in result.cut:
        allowed |= g.descendants(c)
    for eid in g.edges:
        if g.status(eid) != oracle[eid]:
            assert eid in allowed, eid
