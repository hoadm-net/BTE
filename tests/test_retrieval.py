from bte.graph import BJG, Edge
from bte.lattice import S, Sigma
from bte.retrieval import BM25, HashEmbedder, Retriever, cosine, tokenize


def build_graph():
    g = BJG()
    g.add_asserted(Edge(id="job", subject="user", relation="works_at",
                        object="Acme Corp", t_transaction="2026-01-01"))
    g.add_asserted(Edge(id="city", subject="user", relation="lives_in",
                        object="Seattle", t_transaction="2026-02-01"))
    g.add_asserted(Edge(id="ins-src", subject="acme_corp",
                        relation="insurer_of", object="Aetna",
                        t_transaction="2026-01-15"))
    g.add_derived(Edge(id="ins", subject="user", relation="insured_by",
                       object="Aetna", t_transaction="2026-01-15",
                       justification=(frozenset({"job", "ins-src"}),)))
    g.add_asserted(Edge(id="old-city", subject="user", relation="lives_in",
                        object="Portland", t_transaction="2025-01-01",
                        t_valid_end="2026-01-31"))
    return g


def make_retriever(g):
    r = Retriever(g, embedder=HashEmbedder())
    r.index()
    return r


def test_hash_embedder_deterministic_and_normalized():
    e = HashEmbedder()
    a, b = e.embed(["alpha beta"]), e.embed(["alpha beta"])
    assert a == b
    assert abs(cosine(a[0], b[0]) - 1.0) < 1e-9


def test_bm25_ranks_matching_doc_first():
    bm = BM25()
    docs = [tokenize("user works at acme"), tokenize("user lives in seattle")]
    bm.fit(docs)
    scores = bm.scores(tokenize("seattle"))
    assert scores[1] > scores[0]


def test_semantic_channel_finds_lexical_match():
    r = make_retriever(build_graph())
    assert r.semantic("user works at acme corp")[0] == "job"


def test_graph_channel_spreads_to_justification_neighbors():
    r = make_retriever(build_graph())
    ranked = r.graph_channel("does the user work at Acme Corp")
    assert "job" in ranked
    assert "ins" in ranked  # dependent of job via justification


def test_temporal_validity_filter():
    g = build_graph()
    r = make_retriever(g)
    hits = r.retrieve("which city does the user live in",
                      reference_time="2026-03-01")
    ids = [e.id for e in hits]
    assert "city" in ids
    assert "old-city" not in ids  # window closed before reference time


def test_orphaned_invalidation_stays_in_index():
    """An edge invalidated with no successor edge to carry a
    "(previously: X)" annotation (e.g. force_status from BBP
    propagation, or a bare retraction) must stay retrievable - as
    negative evidence, not a live fact (render_fact marks it; see
    test_memory.py) - or the reader can reconstruct the exact
    conclusion that was just retired from surviving raw facts (observed
    on the diagnostic probe, res-d4-ass-upda-adj-hi-r1)."""
    g = build_graph()
    r = make_retriever(g)
    assert "job" in r.semantic("acme corp employer")
    g.force_status("job", Sigma(S.TOP, S.BOT))
    r.index()
    assert "job" in r.semantic("acme corp employer")


def test_directly_superseded_edge_leaves_the_index():
    """When a NEW edge replaces an old one (.supersedes set), the new
    edge's own rendering already carries the "(previously: X)" history
    - the old edge must not ALSO surface independently, or the same
    history would be shown twice."""
    g = build_graph()
    g.add_asserted(Edge(id="new-job", subject="user", relation="works_at",
                        object="Globex", t_transaction="2026-03-01",
                        supersedes="job"))
    g.force_status("job", Sigma(S.TOP, S.BOT))
    r = make_retriever(g)
    assert "job" not in r.semantic("acme corp employer")
    assert "new-job" in r.semantic("acme corp employer")


def test_rrf_prefers_multi_channel_agreement():
    g = build_graph()
    r = make_retriever(g)
    hits = r.retrieve("user works at acme corp", reference_time=None, k=3)
    assert hits and hits[0].id == "job"
