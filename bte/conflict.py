"""Cheap-then-escalate conflict detection with axis classification
(proposed-model.md section 2; contradiction-detection.md).

Cheap path: slot match on (subject, relation) for functional relations,
then temporal-overlap check — applied to asserted AND derived edges, so
inherited-window conflicts are caught (the check Engram-style pipelines
skip). Axis: extraction's is_correction flag when present, else LLM
adjudication. Every decision is logged for the classify(e) reliability
measurement (H5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .graph import BJG, Edge
from .llm import CachedLLM

UPDATE = "update"
CORRECTION = "correction"

ADJUDICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "conflict": {"type": "boolean"},
        "axis": {"type": ["string", "null"], "enum": [UPDATE, CORRECTION, None]},
        "reason": {"type": "string"},
    },
    "required": ["conflict", "axis", "reason"],
    "additionalProperties": False,
}

ADJUDICATION_PROMPT = """Two recorded facts about the same subject and relation:

OLD: {old}
NEW: {new}

Decide:
- conflict: can both be simultaneously true (different objects, overlapping
  validity)? If they are compatible (e.g. multi-valued relation), no conflict.
- axis: if conflict, was the OLD fact wrong from the start (correction),
  or did the world change (update)? null if no conflict."""


@dataclass
class Decision:
    new_edge_id: str
    old_edge_id: str
    conflict: bool
    axis: Optional[str]
    via: str  # "cheap" | "llm"
    inherited: bool  # old edge is derived (its window was inherited)


def windows_overlap(a: Edge, b: Edge) -> bool:
    """Half-open windows; None start = unbounded past, None end = open."""
    def before(end: Optional[str], start: Optional[str]) -> bool:
        return end is not None and start is not None and end <= start
    return not (before(a.t_valid_end, b.t_valid_start)
                or before(b.t_valid_end, a.t_valid_start))


def make_llm_adjudicator(llm: CachedLLM) -> Callable[[Edge, Edge], dict]:
    def adjudicate(new: Edge, old: Edge) -> dict:
        def render(e: Edge) -> str:
            return (f"({e.subject}, {e.relation}, {e.object}) "
                    f"valid [{e.t_valid_start}, {e.t_valid_end}) "
                    f"recorded at {e.t_transaction}")
        return llm.complete_json(
            "You adjudicate factual conflicts in a personal knowledge graph.",
            ADJUDICATION_PROMPT.format(old=render(old), new=render(new)),
            "adjudication",
            ADJUDICATION_SCHEMA,
        )
    return adjudicate


@dataclass
class ConflictDetector:
    adjudicate: Optional[Callable[[Edge, Edge], dict]] = None
    functional: Optional[set[str]] = None  # None = treat all as functional
    log: list[Decision] = field(default_factory=list)
    # relations the adjudicator has repeatedly judged non-conflicting
    # (multi-valued, e.g. watchlist items): skip further checks. On
    # open-domain data, checking every same-slot pair otherwise makes
    # adjudication quadratic in slot size.
    learned_multivalued: set[str] = field(default_factory=set)
    _nonconflict_strikes: dict[str, int] = field(default_factory=dict)
    multivalued_after: int = 2

    def check(self, g: BJG, new_edge: Edge,
              is_correction: bool = False) -> list[Decision]:
        decisions = []
        if (not is_correction
                and new_edge.relation in self.learned_multivalued):
            return decisions
        for old in g.find(subject=new_edge.subject,
                          relation=new_edge.relation):
            if old.object == new_edge.object:
                continue
            if (self.functional is not None
                    and new_edge.relation not in self.functional):
                continue
            if not windows_overlap(old, new_edge):
                continue
            inherited = old.source_type == "derived"
            if is_correction:
                d = Decision(new_edge.id, old.id, True, CORRECTION,
                             "cheap", inherited)
            elif self.adjudicate is not None:
                verdict = self.adjudicate(new_edge, old)
                d = Decision(new_edge.id, old.id, bool(verdict["conflict"]),
                             verdict.get("axis"), "llm", inherited)
                if not d.conflict:
                    rel = new_edge.relation
                    strikes = self._nonconflict_strikes.get(rel, 0) + 1
                    self._nonconflict_strikes[rel] = strikes
                    if strikes >= self.multivalued_after:
                        self.learned_multivalued.add(rel)
            else:
                # functional slot + overlapping window and no adjudicator:
                # recency-wins update is the conservative default
                d = Decision(new_edge.id, old.id, True, UPDATE,
                             "cheap", inherited)
            self.log.append(d)
            if d.conflict:
                decisions.append(d)
        return decisions
