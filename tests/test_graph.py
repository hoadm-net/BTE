import pytest

from bte import BJG, TOP, UNKNOWN, Edge, Sigma, S


def test_insertion_guards():
    g = BJG()
    g.add_asserted(Edge(id="a"))
    with pytest.raises(ValueError):
        g.add_asserted(Edge(id="a"))
    with pytest.raises(ValueError):
        g.add_derived(Edge(id="d", justification=()))
    with pytest.raises(ValueError):
        g.add_derived(Edge(id="d", justification=(frozenset(),)))
    with pytest.raises(ValueError):
        g.add_derived(Edge(id="d", justification=(frozenset({"missing"}),)))
    with pytest.raises(ValueError):
        g.add_derived(Edge(id="d", justification=(frozenset({"d"}),)))
    with pytest.raises(ValueError):
        g.add_asserted(Edge(id="b", justification=(frozenset({"a"}),)))


def test_derived_status_computed_at_insertion():
    g = BJG()
    g.add_asserted(Edge(id="a"), TOP)
    g.add_asserted(Edge(id="b"), UNKNOWN)
    g.add_derived(Edge(id="d1", justification=(frozenset({"a"}),)))
    g.add_derived(Edge(id="d2", justification=(frozenset({"a", "b"}),)))
    g.add_derived(
        Edge(id="d3", justification=(frozenset({"b"}), frozenset({"a"})))
    )
    assert g.status("d1") == TOP
    assert g.status("d2") == UNKNOWN  # meet with the U-status premise
    assert g.status("d3") == TOP  # second alternative suffices


def test_dependents_and_descendants():
    g = BJG()
    g.add_asserted(Edge(id="a"))
    g.add_derived(Edge(id="b", justification=(frozenset({"a"}),)))
    g.add_derived(Edge(id="c", justification=(frozenset({"b"}),)))
    assert g.dependents("a") == {"b"}
    assert g.descendants("a") == {"b", "c"}
    assert g.descendants("c") == set()


def test_status_is_per_axis():
    g = BJG()
    g.add_asserted(Edge(id="a"), Sigma(S.TOP, S.BOT))
    g.add_derived(Edge(id="d", justification=(frozenset({"a"}),)))
    assert g.status("d") == Sigma(S.TOP, S.BOT)
