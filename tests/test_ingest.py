"""LLM-free end-to-end ingestion tests: canned extraction output through
insertion, rule closure, conflict resolution, and propagation — the
Phase 2 exit-criterion smoke test in executable form.
"""

from bte.conflict import ConflictDetector, Decision, windows_overlap
from bte.graph import Edge
from bte.ingest import Ingestor
from bte.lattice import S
from bte.rules import ChainRule


def fact(subject, relation, object, premises=(), correction=False,
         valid_from=None, valid_to=None):
    return {
        "subject": subject, "relation": relation, "object": object,
        "valid_from": valid_from, "valid_to": valid_to,
        "confidence": 1.0, "is_correction": correction,
        "premises": list(premises),
    }


def test_windows_overlap_open_and_closed():
    def edge(start, end):
        return Edge(id="x", t_valid_start=start, t_valid_end=end)
    assert windows_overlap(edge(None, None), edge(None, None))
    assert windows_overlap(edge("2020", None), edge("2024", None))
    assert not windows_overlap(edge("2019", "2021"), edge("2021", None))
    assert windows_overlap(edge("2019", "2022"), edge("2021", None))


def test_correction_propagates_to_stated_consequence():
    """The insurance scenario from the Mem0 smoke test, on our pipeline."""
    ing = Ingestor()
    ing.ingest_facts([fact("user", "works_at", "Acme")], "t1")
    ing.ingest_facts([fact(
        "user", "insured_by", "Aetna",
        premises=[{"subject": "user", "relation": "works_at",
                   "object": "Acme"}],
    )], "t2")
    insured = ing.graph.find(subject="user", relation="insured_by")
    assert len(insured) == 1 and insured[0].source_type == "derived"

    report = ing.ingest_facts(
        [fact("user", "works_at", "Apex Analytics", correction=True)], "t3")

    assert any(d.axis == "correction" for d in report.decisions)
    # old employer superseded on the transaction axis, and the stated
    # consequence (insurance) went down with it
    old = next(e for e in ing.graph.edges.values()
               if e.object == "Acme" and e.relation == "works_at")
    assert ing.graph.status(old.id).trans == S.BOT
    assert not ing.graph.find(subject="user", relation="insured_by")
    assert ing.graph.find(subject="user", relation="works_at")[0].object \
        == "Apex Analytics"


def test_update_via_rule_closure_two_hops():
    """works_at + insurer chain built by rules, then a valid-axis update."""
    rules = [ChainRule("works_at", "insurer_of", "insured_by")]
    ing = Ingestor(rules=rules)
    ing.ingest_facts([
        fact("user", "works_at", "acme"),
        fact("acme", "insurer_of", "Aetna"),
    ], "t1")
    insured = ing.graph.find(subject="user", relation="insured_by")
    assert len(insured) == 1
    assert insured[0].justification  # justified by the two premises

    ing.ingest_facts([fact("user", "works_at", "apex")], "t2")
    # update axis: old employment valid-closed, derived insurance follows
    assert not ing.graph.find(subject="user", relation="insured_by",
                              active_only=True) or \
        ing.graph.find(subject="user", relation="insured_by")[0].id \
        != insured[0].id
    assert ing.graph.status(insured[0].id).valid == S.BOT


def test_multi_path_conclusion_survives_single_premise_loss():
    """Two rule paths to the same conclusion: alternative support."""
    rules = [
        ChainRule("works_at", "insurer_of", "insured_by"),
        ChainRule("member_of", "insurer_of", "insured_by"),
    ]
    ing = Ingestor(rules=rules)
    ing.ingest_facts([
        fact("user", "works_at", "acme"),
        fact("user", "member_of", "acme"),
        fact("acme", "insurer_of", "Aetna"),
    ], "t1")
    insured = ing.graph.find(subject="user", relation="insured_by")
    assert len(insured) == 1
    assert len(insured[0].justification) == 2

    # correcting employment alone must not kill the conclusion
    ing.ingest_facts(
        [fact("user", "works_at", "globex", correction=True)], "t2")
    assert ing.graph.is_active(insured[0].id)


def test_retraction_without_replacement():
    """"I withdrew from the marathon" - the fact is retracted (update
    axis), nothing new asserted, and stated consequences follow it down."""
    ing = Ingestor()
    ing.ingest_facts([fact("user", "training_for", "marathon")], "t1")
    ing.ingest_facts([fact(
        "user", "long_runs_on", "saturday",
        premises=[{"subject": "user", "relation": "training_for",
                   "object": "marathon"}],
    )], "t2")
    report = ing.ingest_facts([], "t3")
    ing.apply_retractions([{
        "subject": "user", "relation": "training_for",
        "object": "marathon", "was_wrong": False,
    }], report)

    assert report.propagations and not report.asserted
    assert not ing.graph.find(subject="user", relation="training_for")
    assert not ing.graph.find(subject="user", relation="long_runs_on")


def test_duplicate_statement_not_reinserted():
    ing = Ingestor()
    ing.ingest_facts([fact("user", "lives_in", "Seattle")], "t1")
    report = ing.ingest_facts([fact("user", "lives_in", "Seattle")], "t2")
    assert not report.asserted
    assert len(ing.graph.edges) == 1


def test_redundant_retraction_spares_replacement():
    """Replacement fact and retraction of the old fact in one batch: the
    retraction must not fall through to the freshly inserted replacement."""
    ing = Ingestor()
    ing.ingest_facts([fact("user", "works_at", "Acme")], "t1")
    report = ing.ingest_facts([fact("user", "works_at", "Apex")], "t2")
    ing.apply_retractions([{
        "subject": "user", "relation": "works_at",
        "object": "Acme", "was_wrong": False,
    }], report)
    active = ing.graph.find(subject="user", relation="works_at")
    assert [e.object for e in active] == ["Apex"]


def test_decisions_are_logged_for_h5():
    det = ConflictDetector()
    ing = Ingestor(detector=det)
    ing.ingest_facts([fact("user", "lives_in", "Seattle")], "t1")
    ing.ingest_facts([fact("user", "lives_in", "Austin")], "t2")
    assert len(det.log) == 1
    d = det.log[0]
    assert isinstance(d, Decision) and d.conflict and d.via == "cheap"


def test_detector_learns_multivalued_relations():
    from bte.graph import BJG, Edge as E
    calls = []

    def fake_adjudicate(new, old):
        calls.append((new.id, old.id))
        return {"conflict": False, "axis": None, "reason": "multi-valued"}

    det = ConflictDetector(adjudicate=fake_adjudicate, multivalued_after=2)
    g = BJG()
    for i, title in enumerate(["Dune", "Arrival", "Contact", "Solaris"]):
        e = E(id=f"w{i}", subject="user", relation="wants_to_watch",
              object=title)
        g.add_asserted(e)
        det.check(g, e)
    # strikes at items 2 and 3 (1 + 2 adjudications), then relation is
    # marked multi-valued and item 4 skips adjudication entirely
    assert "wants_to_watch" in det.learned_multivalued
    assert len(calls) == 3
