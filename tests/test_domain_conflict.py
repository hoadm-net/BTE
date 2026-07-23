"""Domain-typed conflict candidate generation (conflict.py's domain
path): a static commonsense dependency prior between coarse life
domains, extended online from confirmed conflicts, gates which
other-domain facts a new fact is checked against - one batched proposer
call per new fact instead of per-pair adjudication. All LLM-free: fake
proposers with hand-scripted verdicts.
"""

from bte.conflict import ConflictDetector
from bte.domains import DOMAIN_DEPENDENCY_PRIOR, DomainDependencies
from bte.ingest import Ingestor
from bte.lattice import S


def fact(subject, relation, object, domain=None, correction=False):
    return {"subject": subject, "relation": relation, "object": object,
            "domain": domain, "valid_from": None, "valid_to": None,
            "confidence": 1.0, "is_correction": correction, "premises": []}


class ScriptedProposer:
    """Returns the scripted verdicts and records every call's candidate
    list, so tests can assert both call count and candidate gating."""

    def __init__(self, verdicts=None):
        self.verdicts = verdicts or []
        self.calls = []

    def __call__(self, new, candidates):
        self.calls.append((new, list(candidates)))
        out = []
        for v in self.verdicts:
            for i, c in enumerate(candidates):
                if c.object == v["object"]:
                    out.append({"index": i, "axis": v["axis"], "reason": "x"})
        return out


def test_domain_path_disabled_by_default():
    proposer = ScriptedProposer()
    det = ConflictDetector(propose=proposer)  # no domain_deps
    ing = Ingestor(detector=det)
    ing.ingest_facts([fact("user", "limits", "limits commitments",
                           domain="health")], "t1")
    ing.ingest_facts([fact("user", "job", "overnight concierge",
                           domain="work_or_study")], "t2")
    assert proposer.calls == []


def test_dependent_domain_candidate_proposed_and_superseded():
    proposer = ScriptedProposer(
        [{"object": "limits commitments", "axis": "update"}])
    det = ConflictDetector(domain_deps=DomainDependencies(),
                           propose=proposer)
    ing = Ingestor(detector=det)
    ing.ingest_facts([fact("user", "limits", "limits commitments",
                           domain="health")], "t1")
    old = ing.graph.find(subject="user")[0]
    report = ing.ingest_facts([fact("user", "job", "overnight concierge",
                                    domain="work_or_study")], "t2")

    assert any(d.via == "domain" and d.conflict for d in report.decisions)
    assert ing.graph.status(old.id).valid == S.BOT


def test_unrelated_domain_not_in_candidate_set():
    proposer = ScriptedProposer()
    det = ConflictDetector(domain_deps=DomainDependencies(),
                           propose=proposer)
    ing = Ingestor(detector=det)
    # possessions is not in work_or_study's dependency set
    assert "possessions_and_devices" not in \
        DOMAIN_DEPENDENCY_PRIOR["work_or_study"]
    ing.ingest_facts([fact("user", "owns", "a kayak",
                           domain="possessions_and_devices")], "t1")
    ing.ingest_facts([fact("user", "job", "overnight concierge",
                           domain="work_or_study")], "t2")
    # health IS in the set, so the call may happen with other candidates,
    # but the kayak fact must never be offered to the proposer
    for _, candidates in proposer.calls:
        assert all(c.object != "a kayak" for c in candidates)


def test_no_candidates_means_no_proposer_call():
    proposer = ScriptedProposer()
    det = ConflictDetector(domain_deps=DomainDependencies(),
                           propose=proposer)
    ing = Ingestor(detector=det)
    ing.ingest_facts([fact("user", "job", "overnight concierge",
                           domain="work_or_study")], "t1")
    assert proposer.calls == []  # empty graph: nothing to check


def test_untyped_facts_skip_domain_path():
    proposer = ScriptedProposer()
    det = ConflictDetector(domain_deps=DomainDependencies(),
                           propose=proposer)
    ing = Ingestor(detector=det)
    ing.ingest_facts([fact("user", "limits", "limits commitments",
                           domain="health")], "t1")
    ing.ingest_facts([fact("user", "job", "overnight concierge")], "t2")
    assert proposer.calls == []


def test_confirmed_conflict_learns_new_domain_pair():
    """A cross-domain conflict confirmed by ANY path teaches the
    dependency layer a pair beyond the prior, and the pair then gates
    future domain-path checks."""
    deps = DomainDependencies()
    assert "preferences_and_habits" not in deps.affects("family_and_social")

    proposer = ScriptedProposer(
        [{"object": "solo backpacking trips", "axis": "update"}])
    det = ConflictDetector(domain_deps=deps, propose=proposer)
    ing = Ingestor(detector=det)
    # seed the learned pair directly (as if an earlier slot/semantic
    # conflict between these domains had been confirmed)
    deps.record("family_and_social", "preferences_and_habits")
    assert "preferences_and_habits" in deps.affects("family_and_social")

    ing.ingest_facts([fact("user", "prefers", "solo backpacking trips",
                           domain="preferences_and_habits")], "t1")
    old = ing.graph.find(subject="user")[0]
    ing.ingest_facts([fact("user", "family", "has newborn twins",
                           domain="family_and_social")], "t2")
    assert ing.graph.status(old.id).valid == S.BOT


def test_domain_conflicts_recorded_into_learned_counts():
    proposer = ScriptedProposer(
        [{"object": "limits commitments", "axis": "update"}])
    deps = DomainDependencies()
    det = ConflictDetector(domain_deps=deps, propose=proposer)
    ing = Ingestor(detector=det)
    ing.ingest_facts([fact("user", "limits", "limits commitments",
                           domain="health")], "t1")
    ing.ingest_facts([fact("user", "job", "overnight concierge",
                           domain="work_or_study")], "t2")
    assert deps.learned[("work_or_study", "health")] == 1


def test_max_candidates_caps_and_prefers_oldest():
    """Staleness lives in facts recorded long before the new one, so
    the cap keeps the oldest slots (a recency-first cap was observed
    evicting the stale target on real STALE data)."""
    proposer = ScriptedProposer()
    det = ConflictDetector(domain_deps=DomainDependencies(),
                           propose=proposer, domain_max_candidates=2)
    ing = Ingestor(detector=det)
    for i in range(4):
        ing.ingest_facts([fact("user", f"habit{i}", f"habit number {i}",
                               domain="health")], f"t{i}")
    proposer.calls.clear()  # count only the work fact's own call
    ing.ingest_facts([fact("user", "job", "overnight concierge",
                           domain="work_or_study")], "t9")
    assert len(proposer.calls) == 1
    _, candidates = proposer.calls[-1]
    assert [c.object for c in candidates] == \
        ["habit number 0", "habit number 1"]


def test_same_relation_history_deduped_to_most_recent():
    """Only a slot's current value is offered - superseded-then-replaced
    history must not crowd the candidate list."""
    proposer = ScriptedProposer()
    det = ConflictDetector(domain_deps=DomainDependencies(),
                           propose=proposer)
    ing = Ingestor(detector=det)
    # same relation twice: without an adjudicator the slot path treats
    # the second as a recency-wins update, leaving one active fact
    ing.ingest_facts([fact("user", "step_goal", "8,000 steps",
                           domain="health")], "t1")
    ing.ingest_facts([fact("user", "step_goal", "8,500 steps",
                           domain="health")], "t2")
    ing.ingest_facts([fact("user", "job", "overnight concierge",
                           domain="work_or_study")], "t9")
    assert len(proposer.calls) == 1
    _, candidates = proposer.calls[-1]
    assert [c.object for c in candidates] == ["8,500 steps"]


def test_proposal_rejected_by_confirm_stage_leaves_old_fact_active():
    proposer = ScriptedProposer(
        [{"object": "limits commitments", "axis": "update"}])
    det = ConflictDetector(
        adjudicate=lambda new, old: {"conflict": False, "axis": None,
                                     "reason": "compatible"},
        domain_deps=DomainDependencies(), propose=proposer)
    ing = Ingestor(detector=det)
    ing.ingest_facts([fact("user", "limits", "limits commitments",
                           domain="health")], "t1")
    old = ing.graph.find(subject="user")[0]
    ing.ingest_facts([fact("user", "job", "overnight concierge",
                           domain="work_or_study")], "t2")
    assert ing.graph.is_active(old.id)
    assert len(proposer.calls) == 1  # proposed, then vetoed by confirm


def test_pair_already_adjudicated_by_semantic_path_not_reproposed():
    class OneVecEmbedder:
        def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

    proposer = ScriptedProposer()
    det = ConflictDetector(
        adjudicate=lambda new, old: {"conflict": False, "axis": None,
                                     "reason": "x"},
        embedder=OneVecEmbedder(), semantic_threshold=0.5,
        domain_deps=DomainDependencies(), propose=proposer)
    ing = Ingestor(detector=det)
    ing.ingest_facts([fact("user", "limits", "limits commitments",
                           domain="health")], "t1")
    ing.ingest_facts([fact("user", "job", "overnight concierge",
                           domain="work_or_study")], "t2")
    # the semantic path already adjudicated the pair (verdict: no
    # conflict); the domain path must not offer it to the proposer again
    for _, candidates in proposer.calls:
        assert all(c.object != "limits commitments" for c in candidates)


def test_long_candidate_list_split_into_chunks():
    """Proposer recall degrades on a long single-shot list. A target
    near the END of a >chunk_size list must still reach the proposer,
    in its own chunk."""
    proposer = ScriptedProposer(
        [{"object": "target value", "axis": "update"}])
    det = ConflictDetector(domain_deps=DomainDependencies(),
                           propose=proposer, domain_max_candidates=25,
                           domain_chunk_size=10)
    ing = Ingestor(detector=det)
    for i in range(24):
        ing.ingest_facts([fact("user", f"filler{i}", f"filler value {i}",
                               domain="health")], f"t{i:02d}")
    ing.ingest_facts([fact("user", "target_rel", "target value",
                           domain="health")], "t24")
    proposer.calls.clear()
    ing.ingest_facts([fact("user", "job", "overnight concierge",
                           domain="work_or_study")], "t99")
    # 25 candidates / chunk_size 10 -> 3 calls, none exceeding chunk_size
    assert len(proposer.calls) == 3
    assert all(len(cands) <= 10 for _, cands in proposer.calls)
    total_offered = sum(len(cands) for _, cands in proposer.calls)
    assert total_offered == 25
    # the target (chronologically last of the 25, so in the final chunk)
    # was actually handed to the proposer, and got confirmed
    old = ing.graph.find(subject="user", relation="target_rel",
                         active_only=False)[0]
    assert not ing.graph.is_active(old.id)


def test_chunk_indices_remap_to_original_candidate():
    """A proposal's index is chunk-local; the detector must map it back
    to the right candidate, not the wrong one from a different chunk."""
    def confirm_only_target(new, old):
        return {"conflict": old.object == "target value", "axis": "update",
                "reason": "x"}

    class AllIndicesProposer:
        def __call__(self, new, candidates):
            return [{"index": i, "axis": "update", "reason": "x"}
                    for i in range(len(candidates))]

    det = ConflictDetector(adjudicate=confirm_only_target,
                           domain_deps=DomainDependencies(),
                           propose=AllIndicesProposer(),
                           domain_max_candidates=15, domain_chunk_size=5)
    ing = Ingestor(detector=det)
    for i in range(14):
        obj = "target value" if i == 3 else f"filler value {i}"
        ing.ingest_facts([fact("user", f"rel{i}", obj,
                               domain="health")], f"t{i:02d}")
    ing.ingest_facts([fact("user", "job", "overnight concierge",
                           domain="work_or_study")], "t99")
    survivors = {e.relation: e for e in ing.graph.find(subject="user")
                if e.domain == "health"}
    assert "rel3" not in survivors  # exactly the target was superseded
    assert len(survivors) == 13


def test_same_domain_cross_relation_candidates_qualify():
    """Self-loop: implicit conflicts inside one domain across different
    relations have no other detection path (gold-pair audit: investing
    preference vs options-trading habit, both finance)."""
    proposer = ScriptedProposer(
        [{"object": "prefers steady growth, never gambles", "axis": "update"}])
    det = ConflictDetector(domain_deps=DomainDependencies(),
                           propose=proposer)
    ing = Ingestor(detector=det)
    ing.ingest_facts([fact("user", "investing_preference",
                           "prefers steady growth, never gambles",
                           domain="finance_and_resources")], "t1")
    old = ing.graph.find(subject="user")[0]
    ing.ingest_facts([fact("user", "trading_habit",
                           "learning options trading in the evenings",
                           domain="finance_and_resources")], "t2")
    assert not ing.graph.is_active(old.id)


def test_other_domain_is_wildcard_in_both_roles():
    """Untyped facts were invisible to the gate in either role (gold-
    pair audit: a DMV errand implying a car sale; an [other]-typed
    backup habit voided by a machine wipe)."""
    proposer = ScriptedProposer()
    det = ConflictDetector(domain_deps=DomainDependencies(),
                           propose=proposer)
    ing = Ingestor(detector=det)
    ing.ingest_facts([fact("user", "car_ownership", "owns a hatchback",
                           domain="possessions_and_devices")], "t1")
    ing.ingest_facts([fact("user", "backup_habit", "keeps license keys",
                           domain="other")], "t2")
    # new "other" fact saw the possessions candidate
    assert [c.object for c in proposer.calls[-1][1]] == ["owns a hatchback"]
    ing.ingest_facts([fact("user", "computer_wipe", "wiped the machine",
                           domain="possessions_and_devices")], "t3")
    # old "other" fact qualified as candidate for a typed new fact
    offered = [c.object for c in proposer.calls[-1][1]]
    assert "keeps license keys" in offered


def test_same_session_facts_not_offered_to_proposer():
    """One session is one consistent snapshot: a coping plan stated
    alongside a schedule must not be read as invalidating it."""
    proposer = ScriptedProposer()
    det = ConflictDetector(domain_deps=DomainDependencies(),
                           propose=proposer)
    ing = Ingestor(detector=det)
    ing.ingest_facts([
        fact("user", "work_schedule", "overnight shift 11pm-7am",
             domain="work_or_study"),
        fact("user", "sleep_plan", "floating sleep blocks",
             domain="health"),
    ], "t1")
    # both facts share t1: neither may be offered against the other,
    # and with no candidates the proposer is not called at all
    assert proposer.calls == []
    # a later session DOES see the t1 facts as candidates
    ing.ingest_facts([fact("user", "job", "overnight concierge",
                           domain="work_or_study")], "t2")
    assert len(proposer.calls) == 1
    offered = {c.object for _, cands in proposer.calls for c in cands}
    assert "floating sleep blocks" in offered


def test_same_session_facts_skip_semantic_adjudication():
    calls = []

    def counting_adjudicate(new, old):
        calls.append((new.id, old.id))
        return {"conflict": True, "axis": "update", "reason": "x"}

    class OneVecEmbedder:
        def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

    det = ConflictDetector(adjudicate=counting_adjudicate,
                           embedder=OneVecEmbedder(),
                           semantic_threshold=0.5)
    ing = Ingestor(detector=det)
    ing.ingest_facts([
        fact("user", "work_schedule", "overnight shift 11pm-7am"),
        fact("user", "sleep_plan", "floating sleep blocks"),
    ], "t1")
    assert calls == []


def test_affects_merges_prior_and_learned():
    deps = DomainDependencies()
    base = deps.affects("health")
    assert "schedule_and_routine" in base
    deps.record("health", "possessions_and_devices")
    assert "possessions_and_devices" in deps.affects("health")
    # unrelated domain untouched
    assert "possessions_and_devices" not in deps.affects("work_or_study")


def test_record_ignores_same_domain_and_untyped():
    deps = DomainDependencies()
    deps.record("health", "health")
    deps.record(None, "health")
    deps.record("health", None)
    assert deps.learned == {}
