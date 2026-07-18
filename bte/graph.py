"""Bitemporal Justification Graph (proposed-model.md, section 1).

Edges are append-only. Derived edges carry a justification: a non-empty
tuple of alternatives, each a non-empty frozenset of existing edge ids.
Because members must already exist and ids are unique, dependency links
always point from older to newer edges, so the justification structure is
a DAG by construction; `_check_members` guards the two ways that argument
can be violated (self-reference, unknown member).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional

from .lattice import TOP, S, Sigma, evaluate

ASSERTED = "asserted"
DERIVED = "derived"


@dataclass
class Edge:
    id: str
    subject: str = ""
    relation: str = ""
    object: str = ""
    t_valid_start: Optional[str] = None
    t_valid_end: Optional[str] = None  # None = open interval
    t_transaction: Optional[str] = None
    source_type: str = ASSERTED
    confidence: float = 1.0
    justification: tuple[frozenset[str], ...] = field(default_factory=tuple)
    supersedes: Optional[str] = None
    # coarse life-domain type (extraction.DOMAINS); drives domain-level
    # conflict candidate generation in conflict.DomainDependencies
    domain: Optional[str] = None


class BJG:
    def __init__(self) -> None:
        self.edges: dict[str, Edge] = {}
        self._status: dict[str, Sigma] = {}
        self._dependents: dict[str, set[str]] = {}

    # -- construction ------------------------------------------------------

    def add_asserted(self, edge: Edge, status: Sigma = TOP) -> None:
        if edge.id in self.edges:
            raise ValueError(f"duplicate edge id {edge.id!r}")
        if edge.justification:
            raise ValueError("asserted edges cannot carry a justification")
        edge.source_type = ASSERTED
        self.edges[edge.id] = edge
        self._status[edge.id] = status
        self._dependents.setdefault(edge.id, set())

    def add_derived(self, edge: Edge) -> None:
        if edge.id in self.edges:
            raise ValueError(f"duplicate edge id {edge.id!r}")
        self._check_members(edge)
        edge.source_type = DERIVED
        self.edges[edge.id] = edge
        self._dependents.setdefault(edge.id, set())
        for member in self._members(edge):
            self._dependents[member].add(edge.id)
        self._status[edge.id] = self._evaluate(edge.id)

    def _check_members(self, edge: Edge) -> None:
        if not edge.justification:
            raise ValueError("derived edges require a justification")
        for alt in edge.justification:
            if not alt:
                raise ValueError("justification alternative cannot be empty")
            for member in alt:
                if member == edge.id:
                    raise ValueError("edge cannot justify itself")
                if member not in self.edges:
                    raise ValueError(f"unknown justification member {member!r}")

    def add_alternative(self, edge_id: str, alt: frozenset[str]) -> None:
        """Record one more independent derivation path for an existing
        derived edge (multi-justification requirement). New support can
        only raise status, so affected descendants are re-evaluated.
        """
        edge = self.edges[edge_id]
        if edge.source_type != DERIVED:
            raise ValueError("alternatives only apply to derived edges")
        if not alt:
            raise ValueError("justification alternative cannot be empty")
        for member in alt:
            if member == edge_id:
                raise ValueError("edge cannot justify itself")
            if member not in self.edges:
                raise ValueError(f"unknown justification member {member!r}")
            if member in self.descendants(edge_id):
                raise ValueError("alternative would create a justification cycle")
        if alt in edge.justification:
            return
        edge.justification = edge.justification + (alt,)
        for member in alt:
            self._dependents[member].add(edge_id)
        self.recompute({edge_id})

    def recompute(self, seed_ids: set[str]) -> None:
        """Re-evaluate the given derived edges and everything downstream.
        Insertion order is topological (members exist before dependents),
        so a single ordered pass is exact.
        """
        affected = set(seed_ids)
        for eid in seed_ids:
            affected |= self.descendants(eid)
        for eid in self.edges:
            if eid in affected and self.edges[eid].source_type == DERIVED:
                self._status[eid] = self._evaluate(eid)

    # -- status ------------------------------------------------------------

    def status(self, edge_id: str) -> Sigma:
        return self._status[edge_id]

    def force_status(self, edge_id: str, status: Sigma) -> None:
        """Set an edge's status as an external input (a contradiction event
        resolving against this edge). Propagation to dependents is BBP's
        job, not this method's.
        """
        self._status[edge_id] = status

    def _evaluate(self, edge_id: str) -> Sigma:
        edge = self.edges[edge_id]
        return evaluate(
            [self._status[m] for m in alt] for alt in edge.justification
        )

    # -- traversal ---------------------------------------------------------

    def _members(self, edge: Edge) -> set[str]:
        return set().union(*edge.justification) if edge.justification else set()

    def dependents(self, edge_id: str) -> set[str]:
        return set(self._dependents.get(edge_id, ()))

    def descendants(self, edge_id: str) -> set[str]:
        seen: set[str] = set()
        stack = [edge_id]
        while stack:
            for dep in self._dependents.get(stack.pop(), ()):
                if dep not in seen:
                    seen.add(dep)
                    stack.append(dep)
        return seen

    def derived_ids(self) -> Iterator[str]:
        for eid, edge in self.edges.items():
            if edge.source_type == DERIVED:
                yield eid

    # -- queries -----------------------------------------------------------

    def is_active(self, edge_id: str) -> bool:
        """Active = not superseded on either axis (status has no BOT)."""
        s = self._status[edge_id]
        return s.valid != S.BOT and s.trans != S.BOT

    def find(
        self,
        subject: Optional[str] = None,
        relation: Optional[str] = None,
        active_only: bool = True,
    ) -> list[Edge]:
        out = []
        for eid, edge in self.edges.items():
            if subject is not None and edge.subject != subject:
                continue
            if relation is not None and edge.relation != relation:
                continue
            if active_only and not self.is_active(eid):
                continue
            out.append(edge)
        return out
