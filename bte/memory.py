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
from .retrieval import INVALIDATED_PREFIX, Retriever

READER_SYSTEM = (
    "You answer questions about a user from their memory facts.\n"
    "Each fact ends with '(recorded DATE)' - the date the user said it, not "
    "necessarily when the described event happened. Use it against the "
    "current date given below for questions about elapsed time ('how long "
    "ago', 'X days/weeks/months before Y') - compute the actual arithmetic, "
    "do not guess. Facts may also carry change history in parentheses: "
    "'(previously: X, recorded DATE)' means the value was X before being "
    "updated - use it for questions about what changed, or increased vs "
    "decreased. A line starting with 'INVALIDATED:' is NOT a current fact - "
    "it is a conclusion that used to hold but no longer does, shown so you "
    "know NOT to state it or rely on it; if no other fact establishes the "
    "current value, treat the matter as unknown rather than falling back "
    "on the invalidated line. This also applies indirectly: a fact about "
    "some OTHER entity (a plan, a portal, a company...) can remain true on "
    "its own even after an INVALIDATED line shows the fact connecting you "
    "to that entity no longer holds - in that case the other entity's fact "
    "no longer describes YOUR current situation, so do not use it to "
    "answer unless a separate, still-valid fact re-establishes your "
    "connection to that entity.\n"
    "Use ONLY the facts given. If the facts do not determine the answer, "
    "reply exactly: unknown"
)


class MemorySystem(Protocol):
    def ingest_session(self, turns: list[str], timestamp: str) -> None: ...

    def answer(self, question: str,
               reference_time: Optional[str] = None) -> str: ...


def render_fact(e: Edge, graph=None, max_history: int = 3) -> str:
    # an edge retrieved despite being inactive is an orphaned
    # invalidation (Retriever._orphaned_invalidations): no successor
    # exists to carry a "(previously: X)" annotation, so say so
    # directly rather than rendering it as a live fact
    if graph is not None and not graph.is_active(e.id):
        recorded = f", recorded {e.t_transaction}" if e.t_transaction else ""
        return (f"- {INVALIDATED_PREFIX}: {e.subject} | {e.relation} | "
                f"{e.object} - this no longer holds{recorded}; the "
                f"information it was based on has changed")
    window = ""
    if e.t_valid_start or e.t_valid_end:
        window = f" (valid {e.t_valid_start or '...'} to {e.t_valid_end or 'now'})"
    parts = []
    if e.t_transaction:
        parts.append(f"recorded {e.t_transaction}")
    if graph is not None:
        prev_id, hops = e.supersedes, 0
        while prev_id and prev_id in graph.edges and hops < max_history:
            prev = graph.edges[prev_id]
            parts.append(f"previously: {prev.object}"
                         + (f", recorded {prev.t_transaction}"
                            if prev.t_transaction else ""))
            prev_id, hops = prev.supersedes, hops + 1
    history = " (" + "; ".join(parts) + ")" if parts else ""
    return f"- {e.subject} | {e.relation} | {e.object}{window}{history}"


class BJGMemory:
    def __init__(self, ingestor: Ingestor, retriever: Retriever,
                 reader: Callable[[str, str], str],
                 k: int = 12,
                 reader_system: str = READER_SYSTEM) -> None:
        # reader_system is per-benchmark: LongMemEval's protocol expects
        # abstention on unanswerable questions (the default's "reply
        # unknown" clause), STALE's response-generation prompts have no
        # such clause (arXiv 2605.06527 Appendix E.1) - a runner aligns
        # the reader with the benchmark's own protocol, not vice versa.
        self.ingestor = ingestor
        self.retriever = retriever
        self.reader = reader
        self.k = k
        self.reader_system = reader_system

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
        today = f"Current date: {reference_time}\n" if reference_time else ""
        user = f"{today}Facts:\n{rendered}\n\nQuestion: {question}"
        return self.reader(self.reader_system, user).strip()

    # -- LLM-free path for tests and structured replay ---------------------

    def ingest_structured(self, facts: list[dict], retractions: list[dict],
                          timestamp: str) -> None:
        report = self.ingestor.ingest_facts(facts, timestamp)
        self.ingestor.apply_retractions(retractions, report)
        self.retriever.index()
