"""Ingestion orchestrator: extraction -> graph insertion (asserted or
premise-justified derived) -> rule closure -> conflict detection ->
non-destructive supersession -> BBP.

Resolution policy is recency-wins (contradiction-detection.md): the
incoming edge supersedes the old one. A correction lowers the loser's
transaction coordinate; an update lowers its valid coordinate. The loser
stays in the graph (append-only) and BBP restores JTS downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .bbp import BBPResult, bbp
from .conflict import CORRECTION, ConflictDetector, Decision
from .graph import BJG, Edge
from .lattice import S, Sigma
from .rules import ChainRule, derive_closure


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
        max_depth: Optional[int] = None,
        theta: float = 0.0,
        decay_factor: float = 0.9,
    ) -> None:
        self.graph = graph or BJG()
        self.extract = extract
        self.detector = detector or ConflictDetector()
        self.rules = rules or []
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
        context = [
            f"({e.subject}, {e.relation}, {e.object})"
            for e in self.graph.find()
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
        candidates = self.graph.find(subject=r["subject"],
                                     relation=r["relation"])
        exact = [e for e in candidates
                 if e.object.lower() == r["object"].lower()]
        if exact:
            return exact
        # object matches an edge already superseded (e.g. by a replacement
        # fact in the same batch): the retraction is redundant, not a
        # license to hit whatever now occupies the slot
        everything = self.graph.find(subject=r["subject"],
                                     relation=r["relation"],
                                     active_only=False)
        if any(e.object.lower() == r["object"].lower() for e in everything):
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
            confidence=f.get("confidence", 1.0),
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
