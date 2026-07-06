"""MemorySystem protocol and our implementation (harness core, .plan 2.5).

The protocol is the seam every compared system implements in Phase 3
(baseline adapters wrap Zep/Mem0/Engram behind the same two methods), so
benchmark runs vary only the memory module while extraction context,
reader, and scoring stay fixed.

BJGMemory = ingestion pipeline + four-channel retrieval + a fixed reader.
The reader is injected as a callable so tests run LLM-free and the
fixed-reader protocol (LongMemEval-V2 style) is explicit rather than
buried in configuration.
"""

from __future__ import annotations

from typing import Callable, Optional, Protocol

from .graph import Edge
from .ingest import Ingestor
from .retrieval import Retriever

READER_SYSTEM = (
    "You answer questions about a user from their memory facts.\n"
    "Facts may carry a change history in parentheses: "
    "'(previously: X, recorded DATE)' means the value was X before being "
    "updated - use it for questions about what changed, or increased vs "
    "decreased.\n"
    "Use ONLY the facts given. If the facts do not determine the answer, "
    "reply exactly: unknown"
)


class MemorySystem(Protocol):
    def ingest_session(self, turns: list[str], timestamp: str) -> None: ...

    def answer(self, question: str,
               reference_time: Optional[str] = None) -> str: ...


def render_fact(e: Edge, graph=None, max_history: int = 3) -> str:
    window = ""
    if e.t_valid_start or e.t_valid_end:
        window = f" (valid {e.t_valid_start or '...'} to {e.t_valid_end or 'now'})"
    history = ""
    if graph is not None:
        prev_id, hops = e.supersedes, 0
        parts = []
        while prev_id and prev_id in graph.edges and hops < max_history:
            prev = graph.edges[prev_id]
            parts.append(f"previously: {prev.object}"
                         + (f", recorded {prev.t_transaction}"
                            if prev.t_transaction else ""))
            prev_id, hops = prev.supersedes, hops + 1
        if parts:
            history = " (" + "; ".join(parts) + ")"
    return f"- {e.subject} | {e.relation} | {e.object}{window}{history}"


class BJGMemory:
    def __init__(self, ingestor: Ingestor, retriever: Retriever,
                 reader: Callable[[str, str], str],
                 k: int = 12) -> None:
        self.ingestor = ingestor
        self.retriever = retriever
        self.reader = reader
        self.k = k

    # -- MemorySystem -----------------------------------------------------

    def ingest_session(self, turns: list[str], timestamp: str) -> None:
        for turn in turns:
            self.ingestor.ingest_text(turn, timestamp)
        self.retriever.index()

    def answer(self, question: str,
               reference_time: Optional[str] = None) -> str:
        hits = self.retriever.retrieve(question, reference_time, k=self.k)
        if not hits:
            return "unknown"
        rendered = "\n".join(
            render_fact(e, graph=self.ingestor.graph) for e in hits)
        user = f"Facts:\n{rendered}\n\nQuestion: {question}"
        return self.reader(READER_SYSTEM, user).strip()

    # -- LLM-free path for tests and structured replay ---------------------

    def ingest_structured(self, facts: list[dict], retractions: list[dict],
                          timestamp: str) -> None:
        report = self.ingestor.ingest_facts(facts, timestamp)
        self.ingestor.apply_retractions(retractions, report)
        self.retriever.index()
