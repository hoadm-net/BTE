"""State-basis retrieval channel (retrieval.py): similarity channels
are premise-locked - a query phrased in a stale fact's vocabulary never
surfaces the newer cross-domain facts that invalidate its premise. The
channel walks the domain-dependency graph backward from the semantic
channel's top hits and surfaces the subject's current facts in the
domains that bear on them. LLM-free via HashEmbedder (token-overlap
semantics, fully deterministic).
"""

from bte.domains import DomainDependencies
from bte.graph import Edge
from bte.ingest import Ingestor
from bte.retrieval import Retriever


def add(ing, relation, object, domain, ts, subject="user"):
    ing.ingest_facts([{
        "subject": subject, "relation": relation, "object": object,
        "domain": domain, "valid_from": None, "valid_to": None,
        "confidence": 1.0, "is_correction": False, "premises": [],
    }], ts)


def build_premise_locked_graph():
    """Old health fact matches the query lexically; new work facts share
    no tokens with it, so pure similarity can never surface them."""
    ing = Ingestor()
    add(ing, "commitment_limit", "limits daily commitments to manage "
        "shaking symptoms", "health", "2024-06-16")
    add(ing, "hobby", "collects vinyl records", "preferences_and_habits",
        "2024-08-01")
    add(ing, "job_role", "overnight concierge", "work_or_study",
        "2025-02-16")
    add(ing, "shift_hours", "eleven pm to seven am", "work_or_study",
        "2025-02-16")
    return ing


# no "user" token: in this 4-edge toy graph the subject token alone
# would give every edge a nonzero score in every channel, which a
# benchmark-scale graph's top-k would drown out anyway
QUERY = "still limiting daily commitments to manage shaking symptoms"


def test_affected_by_reverses_prior():
    deps = DomainDependencies()
    assert "work_or_study" in deps.affected_by("health")
    assert "health" not in deps.affected_by("possessions_and_devices")
    assert deps.affected_by(None) == frozenset()


def test_affected_by_includes_learned_edges():
    deps = DomainDependencies()
    assert "possessions_and_devices" not in deps.affected_by("health")
    deps.record("possessions_and_devices", "health")
    assert "possessions_and_devices" in deps.affected_by("health")


def test_channel_disabled_without_domain_deps():
    """Channel contract only: no domain_deps, no basis. (The premise-
    locked baseline itself is not reproducible in a toy graph - with a
    handful of edges the temporal channel alone surfaces recent work
    facts, which a benchmark-scale graph's recency top-k drowns out;
    the at-scale evidence lives in the STALE diagnosis, not here.)"""
    ing = build_premise_locked_graph()
    r = Retriever(ing.graph)
    r.index()
    assert r.state_basis(QUERY) == []


def test_channel_surfaces_cross_domain_state():
    ing = build_premise_locked_graph()
    r = Retriever(ing.graph, domain_deps=DomainDependencies())
    r.index()
    basis = r.state_basis(QUERY)
    objects = {ing.graph.edges[eid].object for eid in basis}
    assert "overnight concierge" in objects
    assert "eleven pm to seven am" in objects


def test_fused_retrieval_includes_state_basis_facts():
    ing = build_premise_locked_graph()
    r = Retriever(ing.graph, domain_deps=DomainDependencies())
    r.index()
    hits = r.retrieve(QUERY, k=4)
    assert any(e.domain == "work_or_study" for e in hits)


def test_basis_ranked_by_recency_and_capped():
    ing = Ingestor()
    add(ing, "commitment_limit", "limits daily commitments", "health",
        "2024-06-16")
    for i in range(4):
        add(ing, f"work_fact_{i}", f"work detail {i}", "work_or_study",
            f"2025-01-0{i + 1}")
    r = Retriever(ing.graph, domain_deps=DomainDependencies(),
                  per_channel=3)
    r.index()
    basis = r.state_basis("limits daily commitments")
    objects = [ing.graph.edges[eid].object for eid in basis]
    # anchor's own domain (self-loop) leads, then the work facts newest
    # first, cut at per_channel
    assert objects == ["limits daily commitments",
                       "work detail 3", "work detail 2"]


def test_noisy_domain_cannot_exhaust_channel():
    """Per-domain quota: a domain with many recent facts contributes at
    most basis_per_domain, so an older state change in another source
    domain still surfaces (the global-recency version lost exactly
    this)."""
    ing = Ingestor()
    add(ing, "commitment_limit", "limits daily commitments", "health",
        "2024-06-16")
    for i in range(5):
        add(ing, f"routine_{i}", f"routine detail {i}",
            "schedule_and_routine", f"2025-03-0{i + 1}")
    add(ing, "job_role", "overnight concierge", "work_or_study",
        "2025-01-01")
    r = Retriever(ing.graph, domain_deps=DomainDependencies(),
                  basis_per_domain=2, per_channel=4)
    r.index()
    basis = r.state_basis("limits daily commitments")
    domains = [ing.graph.edges[eid].domain for eid in basis]
    assert domains.count("schedule_and_routine") == 2
    assert "work_or_study" in domains


def test_learned_counts_prioritize_source_domains():
    """Write-time evidence steers query-time recovery: a source domain
    the detector confirmed invalidating the anchor's domain outranks a
    merely-prior source with newer facts."""
    ing = Ingestor()
    add(ing, "commitment_limit", "limits daily commitments", "health",
        "2024-06-16")
    add(ing, "meal_time", "dinner at seven", "schedule_and_routine",
        "2025-01-05")
    add(ing, "job_role", "overnight concierge", "work_or_study",
        "2025-01-01")
    deps = DomainDependencies()
    deps.record("work_or_study", "health")
    r = Retriever(ing.graph, domain_deps=deps)
    r.index()
    basis = r.state_basis("limits daily commitments")
    assert ing.graph.edges[basis[0]].relation == "job_role"


def test_only_same_subject_facts_pulled():
    ing = build_premise_locked_graph()
    add(ing, "job_role", "surgeon", "work_or_study", "2025-03-01",
        subject="partner")
    r = Retriever(ing.graph, domain_deps=DomainDependencies())
    r.index()
    basis = r.state_basis(QUERY)
    subjects = {ing.graph.edges[eid].subject for eid in basis}
    assert subjects == {"user"}


def test_superseded_facts_never_in_basis():
    ing = build_premise_locked_graph()
    # supersede the job fact via the slot path (recency-wins default)
    add(ing, "job_role", "barista", "work_or_study", "2025-04-01")
    r = Retriever(ing.graph, domain_deps=DomainDependencies())
    r.index()
    basis = r.state_basis(QUERY)
    objects = {ing.graph.edges[eid].object for eid in basis}
    assert "overnight concierge" not in objects
    assert "barista" in objects


def test_untyped_seed_contributes_nothing():
    ing = Ingestor()
    ing.graph.add_asserted(Edge(id="e-untyped", subject="user",
                                relation="note", object="limits daily "
                                "commitments", t_transaction="2024-01-01"))
    add(ing, "job_role", "overnight concierge", "work_or_study",
        "2025-02-16")
    r = Retriever(ing.graph, domain_deps=DomainDependencies())
    r.index()
    assert r.state_basis("limits daily commitments") == []
