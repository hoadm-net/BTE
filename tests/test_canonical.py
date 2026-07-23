"""Write-time relation canonicalization (canonical.py): a new fact's
relation is mapped onto the subject's existing relation vocabulary when
they name the same attribute - exact, then loose key, then embedding
auto-merge, then LLM verification for the gray band. All LLM-free via
hand-assigned vectors and scripted verifiers.
"""

from bte.canonical import RelationCanonicalizer, loose_key
from bte.ingest import Ingestor


def fact(subject, relation, object, domain=None):
    return {"subject": subject, "relation": relation, "object": object,
            "domain": domain, "valid_from": None, "valid_to": None,
            "confidence": 1.0, "is_correction": False, "premises": []}


class VecEmbedder:
    """relation text (underscores already spaced) -> assigned vector."""

    def __init__(self, vectors):
        self.vectors = vectors

    def embed(self, texts):
        return [self.vectors.get(t, [0.0, 0.0, 1.0]) for t in texts]


def test_loose_key_normalizes_casing_and_punctuation():
    assert loose_key("Wake_Up-Time") == loose_key("wakeuptime")


def test_exact_and_loose_reuse_without_embedder():
    canon = RelationCanonicalizer()
    ing = Ingestor(canonicalizer=canon)
    ing.ingest_facts([fact("user", "wake_up_time", "7:15 am")], "t1")
    ing.ingest_facts([fact("user", "wakeup_time", "6:30 am")], "t2")
    rels = {e.relation for e in ing.graph.find(subject="user",
                                               active_only=False)}
    assert rels == {"wake_up_time"}
    assert canon.log == [("user", "wakeup_time", "wake_up_time", "loose")]


def test_loose_merge_lets_slot_path_catch_the_update():
    """The point of canonicalization: the update lands in the same slot,
    so the exact-slot path supersedes the old value."""
    ing = Ingestor(canonicalizer=RelationCanonicalizer())
    ing.ingest_facts([fact("user", "wake_up_time", "7:15 am")], "t1")
    old = ing.graph.find(subject="user")[0]
    ing.ingest_facts([fact("user", "Wake-Up_Time", "4:30 pm")], "t2")
    assert not ing.graph.is_active(old.id)
    active = ing.graph.find(subject="user")
    assert [e.object for e in active] == ["4:30 pm"]


def test_embedding_auto_merge_above_threshold():
    emb = VecEmbedder({
        "wake up time": [1.0, 0.0, 0.0],
        "wakes up at": [0.98, 0.02, 0.0],
    })
    canon = RelationCanonicalizer(embedder=emb, auto_threshold=0.9)
    ing = Ingestor(canonicalizer=canon)
    ing.ingest_facts([fact("user", "wake_up_time", "7:15 am")], "t1")
    ing.ingest_facts([fact("user", "wakes_up_at", "6:30 am")], "t2")
    rels = {e.relation for e in ing.graph.find(subject="user",
                                               active_only=False)}
    assert rels == {"wake_up_time"}
    assert canon.log[-1][3] == "embed"


def test_dissimilar_relation_kept_as_new_slot():
    emb = VecEmbedder({
        "wake up time": [1.0, 0.0, 0.0],
        "favorite color": [0.0, 1.0, 0.0],
    })
    canon = RelationCanonicalizer(embedder=emb)
    ing = Ingestor(canonicalizer=canon)
    ing.ingest_facts([fact("user", "wake_up_time", "7:15 am")], "t1")
    ing.ingest_facts([fact("user", "favorite_color", "blue")], "t2")
    rels = {e.relation for e in ing.graph.find(subject="user")}
    assert rels == {"wake_up_time", "favorite_color"}
    assert canon.log == []


def test_gray_band_asks_verifier_and_respects_no():
    emb = VecEmbedder({
        "wake up time": [1.0, 0.0, 0.0],
        "bedtime": [0.85, 0.53, 0.0],  # cosine ~0.85: gray band
    })
    asked = []

    def verifier(a, b):
        asked.append((a, b))
        return False

    canon = RelationCanonicalizer(embedder=emb, auto_threshold=0.95,
                                  verify_threshold=0.7, verify=verifier)
    ing = Ingestor(canonicalizer=canon)
    ing.ingest_facts([fact("user", "wake_up_time", "7:15 am")], "t1")
    ing.ingest_facts([fact("user", "bedtime", "11 pm")], "t2")
    assert len(asked) == 1
    a, b = asked[0]
    assert "bedtime = 11 pm" == a and "wake_up_time = 7:15 am" == b
    rels = {e.relation for e in ing.graph.find(subject="user")}
    assert rels == {"wake_up_time", "bedtime"}  # verifier said no


def test_gray_band_merge_on_verifier_yes():
    emb = VecEmbedder({
        "wake up time": [1.0, 0.0, 0.0],
        "wakes up at": [0.85, 0.53, 0.0],
    })
    canon = RelationCanonicalizer(embedder=emb, auto_threshold=0.95,
                                  verify_threshold=0.7,
                                  verify=lambda a, b: True)
    ing = Ingestor(canonicalizer=canon)
    ing.ingest_facts([fact("user", "wake_up_time", "7:15 am")], "t1")
    ing.ingest_facts([fact("user", "wakes_up_at", "6:30 am")], "t2")
    rels = {e.relation for e in ing.graph.find(subject="user",
                                               active_only=False)}
    assert rels == {"wake_up_time"}
    assert canon.log[-1][3] == "verify"


def test_superseded_relation_names_still_anchor_vocabulary():
    """Names converge to first-seen even when the first fact was
    already replaced (active_only=False lookup)."""
    ing = Ingestor(canonicalizer=RelationCanonicalizer())
    ing.ingest_facts([fact("user", "wake_up_time", "7:15 am")], "t1")
    ing.ingest_facts([fact("user", "wake_up_time", "6:30 am")], "t2")
    ing.ingest_facts([fact("user", "WakeUpTime", "5:00 am")], "t3")
    active = ing.graph.find(subject="user")
    assert [e.relation for e in active] == ["wake_up_time"]


def test_retraction_relation_canonicalized_before_lookup():
    ing = Ingestor(canonicalizer=RelationCanonicalizer())
    ing.ingest_facts([fact("user", "wake_up_time", "7:15 am")], "t1")
    old = ing.graph.find(subject="user")[0]
    report = ing.ingest_facts([], "t2")
    ing.apply_retractions(
        [{"subject": "user", "relation": "Wake_Up_Time",
          "object": "7:15 am", "was_wrong": False}], report)
    assert not ing.graph.is_active(old.id)


def test_different_subjects_do_not_share_vocabulary():
    ing = Ingestor(canonicalizer=RelationCanonicalizer())
    ing.ingest_facts([fact("user", "wake_up_time", "7:15 am")], "t1")
    ing.ingest_facts([fact("partner", "wakeup_time", "8:00 am")], "t2")
    partner = ing.graph.find(subject="partner")
    assert [e.relation for e in partner] == ["wakeup_time"]
