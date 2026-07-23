"""Ingestion orchestrator: extraction -> graph insertion (asserted or
premise-justified derived) -> rule closure -> conflict detection ->
non-destructive supersession -> BBP.

Resolution policy is recency-wins (contradiction-detection.md): the
incoming edge supersedes the old one. A correction lowers the loser's
transaction coordinate; an update lowers its valid coordinate. The loser
stays in the graph (append-only) and BBP restores JTS downstream.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from .bbp import BBPResult, bbp
from .canonical import RelationCanonicalizer
from .conflict import CORRECTION, ConflictDetector, Decision
from .graph import BJG, Edge
from .lattice import S, Sigma
from .rules import ChainRule, derive_closure


def _loose_key(s: str) -> str:
    """Lowercase with all non-alphanumeric characters stripped, so
    'CasCorp', 'cas_corp', and 'Cas Corp' compare equal."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _clamp_confidence(c: object) -> float:
    """Defense in depth alongside the schema's minimum/maximum
    constraint (extraction.py): a confidence outside [0, 1] silently
    poisons every derived edge downstream via derive_closure's min()
    and defeats BBP's theta cutoff (theta=0 is documented as
    "unbounded", but any negative confidence is < 0 by construction,
    truncating propagation after one hop regardless of theta). Found
    via the diagnostic probe: deepseek-v3.2 occasionally emitted -1 for
    confidence on facts about third parties, not the speaker."""
    try:
        c = float(c)
    except (TypeError, ValueError):
        return 1.0
    return min(1.0, max(0.0, c))


@dataclass
class IngestReport:
    asserted: list[str] = field(default_factory=list)
    derived: list[str] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    propagations: list[BBPResult] = field(default_factory=list)


class Ingestor:
    def __init__(
        self,
        graph: Optional[BJG] = None,
        extract: Optional[Callable[[str, str], list[dict]]] = None,
        detector: Optional[ConflictDetector] = None,
        rules: Optional[list[ChainRule]] = None,
        canonicalizer: Optional[RelationCanonicalizer] = None,
        max_depth: Optional[int] = None,
        theta: float = 0.0,
        decay_factor: float = 0.9,
    ) -> None:
        self.graph = graph or BJG()
        self.extract = extract
        self.detector = detector or ConflictDetector()
        self.rules = rules or []
        self.canonicalizer = canonicalizer
        self.max_depth = max_depth
        self.theta = theta
        self.decay_factor = decay_factor
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"e{self._counter}"

    # -- entry points --------------------------------------------------

    def ingest_text(self, text: str, t_transaction: str) -> IngestReport:
        if self.extract is None:
            raise ValueError("no extractor configured")
        # most recent active facts only: on benchmark-scale histories the
        # full graph no longer fits usefully in the extraction context
        context = [
            f"({e.subject}, {e.relation}, {e.object})"
            for e in self.graph.find()[-150:]
        ]
        payload = self.extract(text, t_transaction, context)
        report = self.ingest_facts(payload.get("facts", []), t_transaction)
        self.apply_retractions(payload.get("retractions", []), report)
        return report

    def apply_retractions(self, retractions: list[dict],
                          report: IngestReport) -> None:
        for r in retractions:
            for e in self._retraction_targets(r):
                old_status = self.graph.status(e.id)
                lowered = (Sigma(old_status.valid, S.BOT) if r["was_wrong"]
                           else Sigma(S.BOT, old_status.trans))
                report.propagations.append(bbp(
                    self.graph, e.id, lowered,
                    max_depth=self.max_depth, theta=self.theta,
                    decay_factor=self.decay_factor,
                ))

    def _retraction_targets(self, r: dict) -> list[Edge]:
        if self.canonicalizer is not None:
            r["relation"] = self.canonicalizer.canonical(
                self.graph, r["subject"], r["relation"], r["object"])
        # Match objects up to case/spacing/casing-convention only (e.g.
        # "CasCorp" vs "cas_corp"): the model is told to copy the known
        # triple's object exactly but sometimes reformats it, and an
        # exact-string match failing here means the "already handled"
        # check below silently misses, falling through to the lone-
        # occupant heuristic and retargeting the retraction onto whatever
        # NEW edge happens to occupy the slot (observed: a same-turn
        # replacement fact + a now-redundant retraction of the old value
        # raced, and the retraction ended up superseding the replacement
        # instead of being recognized as already-satisfied).
        target_key = _loose_key(r["object"])
        candidates = self.graph.find(subject=r["subject"],
                                     relation=r["relation"])
        exact = [e for e in candidates if _loose_key(e.object) == target_key]
        if exact:
            return exact
        # object matches an edge already superseded (e.g. by a replacement
        # fact in the same batch): the retraction is redundant, not a
        # license to hit whatever now occupies the slot
        everything = self.graph.find(subject=r["subject"],
                                     relation=r["relation"],
                                     active_only=False)
        if any(_loose_key(e.object) == target_key for e in everything):
            return []
        # a lone slot occupant is unambiguous even if the object spelling
        # drifted between the known-facts rendering and the model's copy
        return candidates if len(candidates) == 1 else []

    def ingest_facts(self, facts: list[dict],
                     t_transaction: str) -> IngestReport:
        report = IngestReport()
        for f in facts:
            self._ingest_one(f, t_transaction, report)
        derived = derive_closure(
            self.graph, self.rules, self._next_id, t_transaction)
        report.derived += derived
        for eid in derived:
            self._resolve(self.graph.edges[eid], False, report)
        return report

    # -- internals ------------------------------------------------------

    def _find_premise(self, p: dict) -> Optional[str]:
        for e in self.graph.find(subject=p["subject"],
                                 relation=p["relation"]):
            if e.object.lower() == p["object"].lower():
                return e.id
        return None

    def _ingest_one(self, f: dict, t_transaction: str,
                    report: IngestReport) -> None:
        if self.canonicalizer is not None:
            f["relation"] = self.canonicalizer.canonical(
                self.graph, f["subject"], f["relation"], f["object"])
            for p in f.get("premises") or ():
                p["relation"] = self.canonicalizer.canonical(
                    self.graph, p["subject"], p["relation"], p["object"])
        duplicates = [
            e for e in self.graph.find(subject=f["subject"],
                                       relation=f["relation"])
            if e.object.lower() == f["object"].lower()
        ]
        if duplicates:
            return  # re-statement of an already-active identical fact
        edge = Edge(
            id=self._next_id(),
            subject=f["subject"],
            relation=f["relation"],
            object=f["object"],
            t_valid_start=f.get("valid_from"),
            t_valid_end=f.get("valid_to"),
            t_transaction=t_transaction,
            confidence=_clamp_confidence(f.get("confidence", 1.0)),
            domain=f.get("domain"),
        )
        premise_ids = [pid for p in f.get("premises", ())
                       if (pid := self._find_premise(p)) is not None]
        if premise_ids:
            edge.justification = (frozenset(premise_ids),)
            self.graph.add_derived(edge)
            report.derived.append(edge.id)
        else:
            self.graph.add_asserted(edge)
            report.asserted.append(edge.id)
        self._resolve(edge, bool(f.get("is_correction")), report)

    def _resolve(self, new_edge: Edge, is_correction: bool,
                 report: IngestReport) -> None:
        decisions = self.detector.check(self.graph, new_edge, is_correction)
        report.decisions += decisions
        for d in decisions:
            # Superseding a derived edge (inherited conflict) forces its
            # status as a resolution input even though its premises still
            # evaluate active; a later recompute() of that edge would
            # resurrect it. Tracked as an open item for 2.3+: root-cause
            # resolution should walk to the premise that made it wrong.
            loser = self.graph.edges[d.old_edge_id]
            old_status = self.graph.status(loser.id)
            if d.axis == CORRECTION:
                lowered = Sigma(old_status.valid, S.BOT)
            else:
                lowered = Sigma(S.BOT, old_status.trans)
            new_edge.supersedes = loser.id
            report.propagations.append(bbp(
                self.graph, loser.id, lowered,
                max_depth=self.max_depth, theta=self.theta,
                decay_factor=self.decay_factor,
            ))
