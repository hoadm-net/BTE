"""Semantic candidate generation in ConflictDetector: catches implicit
conflicts across DIFFERENT relations that same-slot matching never
compares (see conflict.py's module docstring; diagnosed on STALE T2 -
"works overnight at a hotel" vs "needs to limit daily commitments" share
no relation, so the exact-slot path never even considers the pair).
Uses a deterministic fake embedder (hand-assigned vectors) so cosine
similarity is fully controlled and LLM-free.
"""

from bte.conflict import ConflictDetector
from bte.ingest import Ingestor
from bte.lattice import S


def fact(subject, relation, object, correction=False):
    return {"subject": subject, "relation": relation, "object": object,
            "valid_from": None, "valid_to": None, "confidence": 1.0,
            "is_correction": correction, "premises": []}


class FakeEmbedder:
    """text -> hand-assigned vector, by substring match, so tests fully
    control which pairs look similar."""

    def __init__(self, vectors: dict[str, list[float]],
                default=(0.0, 0.0, 1.0)):
        self.vectors = vectors
        self.default = list(default)

    def embed(self, texts):
        out = []
        for t in texts:
            v = self.default
            for key, vec in self.vectors.items():
                if key in t:
                    v = vec
                    break
            out.append(v)
        return out


def fake_adjudicate_always_conflict(new, old):
    return {"conflict": True, "axis": "update", "reason": "semantic"}


def fake_adjudicate_never_conflict(new, old):
    return {"conflict": False, "axis": None, "reason": "unrelated"}


def test_semantic_path_disabled_by_default():
    det = ConflictDetector(adjudicate=fake_adjudicate_always_conflict)
    ing = Ingestor(detector=det)
    ing.ingest_facts([fact("user", "job", "hotel overnight concierge")], "t1")
    ing.ingest_facts(
        [fact("user", "commitment_limit", "needs to limit commitments")],
        "t2")
    active = ing.graph.find(subject="user")
    assert len(active) == 2, "no embedder configured: nothing superseded"


def test_semantic_candidate_escalated_and_conflict_applied():
    embedder = FakeEmbedder({
        "hotel overnight": [1.0, 0.0, 0.0],
        "needs to limit commitments": [0.9, 0.1, 0.0],  # close to above
    })
    det = ConflictDetector(adjudicate=fake_adjudicate_always_conflict,
                           embedder=embedder, semantic_threshold=0.5)
    ing = Ingestor(detector=det)
    ing.ingest_facts(
        [fact("user", "commitment_limit", "needs to limit commitments")],
        "t1")
    old = ing.graph.find(subject="user")[0]
    report = ing.ingest_facts(
        [fact("user", "job", "hotel overnight concierge")], "t2")

    assert any(d.via == "semantic" for d in report.decisions)
    assert ing.graph.status(old.id).valid == S.BOT


def test_below_threshold_candidate_not_escalated():
    embedder = FakeEmbedder({
        "hotel overnight": [1.0, 0.0, 0.0],
        "unrelated hobby": [-1.0, 0.0, 0.0],  # opposite direction
    })
    det = ConflictDetector(adjudicate=fake_adjudicate_always_conflict,
                           embedder=embedder, semantic_threshold=0.5)
    ing = Ingestor(detector=det)
    ing.ingest_facts([fact("user", "hobby", "unrelated hobby")], "t1")
    old = ing.graph.find(subject="user")[0]
    ing.ingest_facts([fact("user", "job", "hotel overnight concierge")], "t2")

    assert ing.graph.is_active(old.id), "dissimilar candidate must not be touched"


def test_same_relation_excluded_from_semantic_path():
    """Same-relation pairs go through the exact-slot path only; the
    semantic path must not double-adjudicate them."""
    calls = []

    def counting_adjudicate(new, old):
        calls.append((new.id, old.id))
        return {"conflict": True, "axis": "update", "reason": "x"}

    embedder = FakeEmbedder({"acme": [1.0, 0.0, 0.0]}, default=[1.0, 0.0, 0.0])
    det = ConflictDetector(adjudicate=counting_adjudicate, embedder=embedder,
                           semantic_threshold=0.0)
    ing = Ingestor(detector=det)
    ing.ingest_facts([fact("user", "works_at", "Acme")], "t1")
    ing.ingest_facts([fact("user", "works_at", "Globex")], "t2")

    # exactly one adjudication (the exact-slot pair) - not a second one
    # from the semantic path re-finding the same pair
    assert len(calls) == 1


def test_non_conflicting_semantic_candidate_leaves_both_active():
    embedder = FakeEmbedder({
        "hotel overnight": [1.0, 0.0, 0.0],
        "favorite color is blue": [0.9, 0.1, 0.0],
    })
    det = ConflictDetector(adjudicate=fake_adjudicate_never_conflict,
                           embedder=embedder, semantic_threshold=0.5)
    ing = Ingestor(detector=det)
    ing.ingest_facts([fact("user", "preference", "favorite color is blue")],
                     "t1")
    old = ing.graph.find(subject="user")[0]
    ing.ingest_facts([fact("user", "job", "hotel overnight concierge")], "t2")

    assert ing.graph.is_active(old.id)


def test_top_k_limits_adjudication_calls():
    calls = []

    def counting_adjudicate(new, old):
        calls.append(old.id)
        return {"conflict": False, "axis": None, "reason": "x"}

    # 5 candidates, pairwise-orthogonal to each other (so ingesting one
    # never triggers a semantic match against an earlier one), but all
    # moderately similar to the final query vector - isolates top_k's
    # effect on the query's OWN candidate ranking.
    vectors = {f"cand{i}": v for i, v in enumerate([
        [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0], [0, 0, 0, 0, 1],
    ])}
    vectors["query"] = [0.5, 0.5, 0.5, 0.5, 0.5]
    embedder = FakeEmbedder(vectors, default=[0, 0, 0, 0, 0])
    det = ConflictDetector(adjudicate=counting_adjudicate, embedder=embedder,
                           semantic_threshold=0.3, semantic_top_k=2)
    ing = Ingestor(detector=det)
    for i in range(5):
        ing.ingest_facts([fact("user", f"rel{i}", f"cand{i}")], f"t{i}")
    calls.clear()  # only count adjudications triggered by the query fact
    ing.ingest_facts([fact("user", "query_rel", "query")], "t9")

    assert len(calls) == 2
