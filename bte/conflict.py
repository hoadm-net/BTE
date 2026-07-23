"""Cheap-then-escalate conflict detection with axis classification
(proposed-model.md section 2; contradiction-detection.md).

Cheap path: slot match on (subject, relation) for functional relations,
then temporal-overlap check — applied to asserted AND derived edges, so
inherited-window conflicts are caught (the check Engram-style pipelines
skip). Axis: extraction's is_correction flag when present, else LLM
adjudication. Every decision is logged for the classify(e) reliability
measurement (H5).

Same-relation matching alone misses *implicit* conflicts where a new
fact contradicts an existing one under a different relation entirely
(observed on STALE T2: "works overnight at a hotel" never gets compared
against "needs to limit daily commitments" — no shared slot, so the
pair is never even escalated, and BBP has nothing to propagate because
no conflict was ever detected upstream of it). The semantic candidate
path below widens "cheap" from exact slot match to hybrid-retrieval-
style top-k similarity search across the subject's other active facts,
still escalating to the same LLM adjudicator used above — no new model,
no training, grounded in the same two-stage retrieve-then-classify
pattern used for open-KB contradiction detection (Chen et al., *A
Straightforward Pipeline for Targeted Entailment and Contradiction
Detection*, 2025, https://arxiv.org/abs/2508.17127: attention/embedding-
based candidate retrieval feeding an off-the-shelf NLI-style classifier,
no fine-tuning). Multi-hop expansion (beam search over the candidate
graph) is deliberately deferred: measure what single-hop retrieval
closes on STALE T2 before adding that complexity.

Embedding similarity, however, cannot see conflicts whose only link is
commonsense: a cosine-based candidate search reliably surfaces conflicts
that share vocabulary, but conflicts connected purely through
real-world implication (no fact restates the connecting concept) sit
far below any threshold that would also catch the lexical ones without
swamping the adjudicator with noise — MemStrata, arXiv 2606.26511,
independently reports AUROC 0.59 (near chance) for cosine on
contradiction-vs-duplicate. The domain path fixes this the
way working 2026 systems converge on: represent dependency structurally
at the TYPE level, not discover it pairwise at the instance level.
Facts carry a coarse life-domain type (extraction.DOMAINS); a small
directed domain-dependency graph (commonsense prior, ~35 edges) gates
which OTHER-domain facts a new fact could implicitly invalidate, and
one batched proposer call per new fact replaces per-pair adjudication.
Two deltas over CUPMem's fixed schema + per-call extrapolation (STALE,
arXiv 2605.06527, Appendix F): the dependency graph LEARNS online —
every confirmed cross-domain conflict (from any path) strengthens that
domain pair, so coverage grows past the prior — and confirmed conflicts
feed the same supersession + BBP machinery, whose propagation is proven
(formalism.md Theorem 2) rather than re-judged per query.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .domains import DomainDependencies
from .graph import BJG, Edge
from .llm import CachedLLM
from .retrieval import Embedder, cosine, edge_text

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

PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "conflicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "axis": {"type": "string", "enum": [UPDATE, CORRECTION]},
                    "reason": {"type": "string"},
                },
                "required": ["index", "axis", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["conflicts"],
    "additionalProperties": False,
}

PROPOSAL_PROMPT = """A new fact about a subject was just recorded. Below it is a numbered list of older facts about the same subject from related life areas.

NEW: {new}

OLDER FACTS:
{candidates}

List ONLY the older facts that CANNOT still be true given the NEW fact:
keeping both as currently-true would contradict how the world works.
Include implicit contradictions - the NEW fact's real-world consequences
(a changed job, schedule, location, health situation...) can rule an
older fact out even when the two share no wording, e.g. a new overnight
work schedule rules out an established early-morning routine.

These are NOT conflicts - never list them:
- a plan, method, or coping strategy for a situation, vs that situation
  (advice about handling a schedule does not invalidate the schedule)
- a one-off event vs a general routine or habit
- plans, purchases, activities, or preferences that can coexist
- earlier steps or variants of the same ongoing activity or project
- facts that merely touch the same topic

For each genuine conflict give its index and the axis: "correction" if
the older fact was never true, "update" if the world changed. If you
are not certain a pair is contradictory, leave it out. Return an empty
list if none conflict."""

ADJUDICATION_PROMPT = """Two recorded facts about the same subject (possibly different relations):

OLD: {old}
NEW: {new}

Decide:
- conflict: can both be simultaneously true? Reason in two steps.
  First, what situation does the NEW fact imply about the subject's
  life - their schedule, location, possessions, obligations, habits?
  Second, is the OLD fact still possible or plausible in that
  situation? It IS a conflict when keeping the OLD fact as
  currently-true would be impossible or clearly implausible given that
  situation, even if the two facts mention entirely different topics:
  selling a car voids its insurance arrangement; switching a driver's
  license to a new address means the subject left the old city; a
  fully-stacked overnight schedule means an earlier practice of
  keeping days lightly committed is no longer being maintained. "They
  describe different aspects of life" is NOT a reason for
  compatibility - different aspects of one life constrain each other.
  If both facts genuinely can hold at once (multi-valued relation,
  independent activities), no conflict.
- axis: if conflict, was the OLD fact wrong from the start (correction),
  or did the world change (update)? null if no conflict."""


@dataclass
class Decision:
    new_edge_id: str
    old_edge_id: str
    conflict: bool
    axis: Optional[str]
    via: str  # "cheap" | "llm" | "semantic" | "domain"
    inherited: bool  # old edge is derived (its window was inherited)


def windows_overlap(a: Edge, b: Edge) -> bool:
    """Half-open windows; None start = unbounded past, None end = open."""
    def before(end: Optional[str], start: Optional[str]) -> bool:
        return end is not None and start is not None and end <= start
    return not (before(a.t_valid_end, b.t_valid_start)
                or before(b.t_valid_end, a.t_valid_start))


def render_edge(e: Edge) -> str:
    return (f"({e.subject}, {e.relation}, {e.object}) "
            f"valid [{e.t_valid_start}, {e.t_valid_end}) "
            f"recorded at {e.t_transaction}")


def make_llm_adjudicator(llm: CachedLLM) -> Callable[[Edge, Edge], dict]:
    def adjudicate(new: Edge, old: Edge) -> dict:
        return llm.complete_json(
            "You adjudicate factual conflicts in a personal knowledge graph.",
            ADJUDICATION_PROMPT.format(old=render_edge(old),
                                       new=render_edge(new)),
            "adjudication",
            ADJUDICATION_SCHEMA,
        )
    return adjudicate


def make_llm_proposer(llm: CachedLLM) -> Callable[[Edge, list[Edge]], list[dict]]:
    """One call covering the whole domain-gated candidate set (vs one
    adjudication per pair on the semantic path)."""
    def propose(new: Edge, candidates: list[Edge]) -> list[dict]:
        listing = "\n".join(f"{i}. {render_edge(e)}"
                            for i, e in enumerate(candidates))
        data = llm.complete_json(
            "You maintain consistency of a personal knowledge graph.",
            PROPOSAL_PROMPT.format(new=render_edge(new), candidates=listing),
            "conflict_proposal",
            PROPOSAL_SCHEMA,
        )
        out = []
        for c in data.get("conflicts") or []:
            i = c.get("index")
            if isinstance(i, int) and 0 <= i < len(candidates):
                out.append(c)
        return out
    return propose


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
    # semantic candidate generation (see module docstring): None disables
    # it entirely (existing same-slot behavior, unchanged).
    embedder: Optional[Embedder] = None
    semantic_top_k: int = 8
    semantic_threshold: float = 0.5
    # domain-typed candidate generation (see module docstring): both
    # domain_deps and propose must be set to enable it.
    domain_deps: Optional[DomainDependencies] = None
    propose: Optional[Callable[[Edge, list[Edge]], list[dict]]] = None
    domain_max_candidates: int = 60
    # proposer recall degrades on a long single-shot list - a candidate
    # near the end of a full-length list can go unproposed even though
    # the same pair confirms positively when adjudicated in isolation -
    # so chunk the list, keeping any single call's attention span short
    domain_chunk_size: int = 20

    def check(self, g: BJG, new_edge: Edge,
              is_correction: bool = False) -> list[Decision]:
        log_start = len(self.log)
        decisions = []
        if not (not is_correction
                and new_edge.relation in self.learned_multivalued):
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
                    d = Decision(new_edge.id, old.id,
                                bool(verdict["conflict"]),
                                verdict.get("axis"), "llm", inherited)
                    if not d.conflict:
                        rel = new_edge.relation
                        strikes = self._nonconflict_strikes.get(rel, 0) + 1
                        self._nonconflict_strikes[rel] = strikes
                        if strikes >= self.multivalued_after:
                            self.learned_multivalued.add(rel)
                else:
                    # functional slot + overlapping window, no
                    # adjudicator: recency-wins update is the
                    # conservative default
                    d = Decision(new_edge.id, old.id, True, UPDATE,
                                 "cheap", inherited)
                self.log.append(d)
                if d.conflict:
                    decisions.append(d)

        if self.embedder is not None and self.adjudicate is not None:
            decisions += self._check_semantic(g, new_edge)
        if self.domain_deps is not None and self.propose is not None:
            # skip pairs already adjudicated this call (any verdict), so
            # the two paths never double-charge the same pair
            seen = {d.old_edge_id for d in self.log[log_start:]}
            decisions += self._check_domain(g, new_edge, seen)
        if self.domain_deps is not None:
            for d in decisions:
                old = g.edges[d.old_edge_id]
                self.domain_deps.record(new_edge.domain, old.domain)
        return decisions

    def _check_semantic(self, g: BJG, new_edge: Edge) -> list[Decision]:
        candidates = [e for e in g.find(subject=new_edge.subject)
                     if e.relation != new_edge.relation
                     and e.t_transaction != new_edge.t_transaction
                     and windows_overlap(e, new_edge)]
        if not candidates:
            return []
        vecs = self.embedder.embed(
            [edge_text(e) for e in candidates] + [edge_text(new_edge)])
        query_vec = vecs[-1]
        ranked = sorted(zip(candidates, vecs[:-1]),
                        key=lambda cv: -cosine(query_vec, cv[1]))
        decisions = []
        for old, vec in ranked[:self.semantic_top_k]:
            if cosine(query_vec, vec) < self.semantic_threshold:
                break
            verdict = self.adjudicate(new_edge, old)
            d = Decision(new_edge.id, old.id, bool(verdict["conflict"]),
                        verdict.get("axis"), "semantic",
                        old.source_type == "derived")
            self.log.append(d)
            if d.conflict:
                decisions.append(d)
        return decisions

    def _check_domain(self, g: BJG, new_edge: Edge,
                      skip: set[str]) -> list[Decision]:
        affected = self.domain_deps.affects(new_edge.domain)
        if not affected:
            return []
        # Same-session facts (equal t_transaction) are excluded from BOTH
        # implicit paths: one session describes one consistent snapshot,
        # and a coping plan stated alongside a schedule must not be read
        # as invalidating it (observed: session 36's "floating sleep
        # blocks" plan superseding that same session's overnight-job
        # facts). Staleness is a cross-session phenomenon; same-slot
        # same-session replacement stays the exact-slot path's job.
        candidates = [e for e in g.find(subject=new_edge.subject)
                      if e.id not in skip
                      and e.relation != new_edge.relation
                      and e.t_transaction != new_edge.t_transaction
                      and e.domain in affected
                      and windows_overlap(e, new_edge)]
        if not candidates:
            return []
        # One representative per relation: the most recent active fact
        # in a slot is the current state a new fact could invalidate;
        # same-relation history adds prompt length, not coverage. The
        # cap then keeps the OLDEST slots first - staleness lives in
        # facts recorded long before the new one (a recency-first cap
        # was observed evicting exactly the stale target this path
        # exists to catch, while an age cap misses nothing that a
        # 60-slot state summary would hold).
        by_relation: dict[str, Edge] = {}
        for e in candidates:
            cur = by_relation.get(e.relation)
            if cur is None or (e.t_transaction or "") > \
                    (cur.t_transaction or ""):
                by_relation[e.relation] = e
        candidates = sorted(by_relation.values(),
                            key=lambda e: e.t_transaction or "")
        candidates = candidates[:self.domain_max_candidates]
        proposals: list[tuple[Edge, str]] = []
        for start in range(0, len(candidates), self.domain_chunk_size):
            chunk = candidates[start:start + self.domain_chunk_size]
            for verdict in self.propose(new_edge, chunk):
                proposals.append((chunk[verdict["index"]],
                                  verdict.get("axis")))
        decisions = []
        for old, axis in proposals:
            # two-stage: the batched proposer is recall-oriented; each
            # proposal is confirmed by the pairwise adjudicator before
            # it supersedes anything (a handful of extra calls per
            # session, and it filters the one-off-event-vs-routine
            # over-triggering observed from the proposer alone)
            if self.adjudicate is not None:
                confirm = self.adjudicate(new_edge, old)
                if not confirm.get("conflict"):
                    continue
                axis = confirm.get("axis") or axis
            d = Decision(new_edge.id, old.id, True, axis,
                         "domain", old.source_type == "derived")
            self.log.append(d)
            decisions.append(d)
        return decisions
