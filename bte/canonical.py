"""Write-time relation-name canonicalization.

Open-domain extraction names relations freely, and the same attribute
drifts across surface forms ("wakes_up_at" / "wake_up_time" /
"wakes_up_time") - within a single run, not only across runs. Every
slot-keyed mechanism downstream silently degrades: the exact-slot
conflict path never sees the update pair, per-relation candidate dedup
treats one attribute as several, and rule matching requires exact
equality. The schema-enum answer (extraction.py, Track A) only exists
when the target vocabulary is known in advance.

For the open-domain case this module follows the two precedents the
extraction docstring already names: EDC's canonicalization phase -
embedding similarity proposes a merge target, an LLM verifies it
(Zhang et al., EMNLP 2024, https://arxiv.org/abs/2404.03868) - and
Zep/Graphiti's write-time edge dedup against the entity's existing
edges. Matching is per subject and cheapest-first: exact name, then
loose key (casing/spacing/punctuation), then embedding cosine over the
subject's known relation names with an auto-merge band, then a single
small LLM verification call for the gray band. New-relation events are
rare after the first sessions, so the added cost is a handful of calls
per scenario, all cached. Every merge is logged (subject, from, to,
via) so naming convergence is measurable, not assumed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from .graph import BJG
from .llm import CachedLLM
from .retrieval import Embedder, cosine

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "same_attribute": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["same_attribute", "reason"],
    "additionalProperties": False,
}

VERIFY_PROMPT = """Two recorded attributes of the same subject:

A: {a}
B: {b}

Do A and B name the SAME underlying attribute of the subject - such
that a new value for one should replace the current value of the other?
Different aspects that merely relate (e.g. wake-up time vs bedtime, or
a goal vs a habit) are NOT the same attribute."""


def loose_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def make_llm_relation_verifier(llm: CachedLLM) -> Callable[[str, str], bool]:
    def verify(a: str, b: str) -> bool:
        data = llm.complete_json(
            "You maintain a consistent relation vocabulary in a "
            "personal knowledge graph.",
            VERIFY_PROMPT.format(a=a, b=b),
            "relation_match",
            VERIFY_SCHEMA,
        )
        return bool(data.get("same_attribute"))
    return verify


@dataclass
class RelationCanonicalizer:
    """Maps a new fact's relation onto the subject's existing relation
    vocabulary when they name the same attribute. Embedder-less
    instances still do exact/loose matching."""

    embedder: Optional[Embedder] = None
    verify: Optional[Callable[[str, str], bool]] = None
    auto_threshold: float = 0.90
    verify_threshold: float = 0.70
    # (subject, original, canonical, via) for every merge performed
    log: list[tuple[str, str, str, str]] = field(default_factory=list)

    def canonical(self, g: BJG, subject: str, relation: str,
                  object: str = "") -> str:
        # all names ever used for this subject, superseded included:
        # naming should converge to first-seen even after replacement
        rels = {e.relation for e in g.find(subject=subject,
                                           active_only=False)}
        if not rels or relation in rels:
            return relation
        lk = loose_key(relation)
        for r in sorted(rels):
            if loose_key(r) == lk:
                self.log.append((subject, relation, r, "loose"))
                return r
        if self.embedder is None:
            return relation
        ordered = sorted(rels)
        texts = [r.replace("_", " ") for r in ordered] \
            + [relation.replace("_", " ")]
        vecs = self.embedder.embed(texts)
        new_vec = vecs[-1]
        best, best_sim = None, -1.0
        for r, v in zip(ordered, vecs[:-1]):
            sim = cosine(new_vec, v)
            if sim > best_sim:
                best, best_sim = r, sim
        if best is None or best_sim < self.verify_threshold:
            return relation
        if best_sim >= self.auto_threshold:
            self.log.append((subject, relation, best, "embed"))
            return best
        if self.verify is not None:
            example = next(
                (e.object for e in g.find(subject=subject, relation=best,
                                          active_only=False)), "")
            if self.verify(f"{relation} = {object}", f"{best} = {example}"):
                self.log.append((subject, relation, best, "verify"))
                return best
        return relation
