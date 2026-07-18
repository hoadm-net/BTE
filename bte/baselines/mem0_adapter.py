"""Mem0 behind the MemorySystem protocol (.plan 03 baseline).

Mem0 does its own extraction and its own ADD/UPDATE/DELETE conflict
handling; we wrap only its memory. To isolate the memory module, answers
go through the SAME fixed reader as our system, fed Mem0's retrieved
memories in the same prompt shape. This is the fixed-reader protocol:
only the memory component differs between systems under comparison.

Each question gets a fresh isolated Mem0 store (per-user collection) so
runs do not leak across questions and can go in parallel.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from typing import Callable, Optional

# Mem0 also opens a fixed-path telemetry vector store at import/init time
# (~/.mem0/migrations_<provider>) regardless of the config passed to
# Memory.from_config, which lock-collides across concurrent instances
# (observed: RuntimeError from qdrant_local under --workers > 1). Disabling
# telemetry is the documented way to skip that store entirely, not a
# workaround for our isolation - the shared path is Mem0's own design, not
# something our config controls.
os.environ.setdefault("MEM0_TELEMETRY", "False")

from mem0 import Memory  # noqa: E402

from ..memory import READER_SYSTEM  # noqa: E402


class Mem0Memory:
    def __init__(self, reader: Callable[[str, str], str], k: int = 12,
                 model: str = "gpt-4o-mini",
                 reader_system: str = READER_SYSTEM) -> None:
        store = tempfile.mkdtemp(prefix="mem0-")
        self._mem = Memory.from_config({
            "vector_store": {
                "provider": "qdrant",
                "config": {"path": store, "collection_name": "bench"},
            },
            "llm": {"provider": "openai",
                    "config": {"model": model, "temperature": 0}},
        })
        self._user = f"u-{uuid.uuid4().hex[:8]}"
        self.reader = reader
        self.k = k
        # same per-benchmark reader configuration as BJGMemory: the
        # fixed-reader protocol fixes the reader ACROSS systems within a
        # benchmark, and each benchmark's runner supplies its protocol's
        # prompt (STALE has no abstention clause, LongMemEval expects one)
        self.reader_system = reader_system

    def ingest_session(self, turns: list[str], timestamp: str) -> None:
        for turn in turns:
            # Mem0 has no valid-time axis; fold the session date into the
            # text so its own extractor can use it, matching how a Mem0
            # user would supply dated turns.
            self._mem.add(f"[{timestamp}] {turn}", user_id=self._user)

    def answer(self, question: str,
               reference_time: Optional[str] = None) -> str:
        hits = self._mem.search(question, filters={"user_id": self._user})
        results = hits.get("results", hits) if isinstance(hits, dict) else hits
        memories = [h.get("memory", str(h)) for h in results][:self.k]
        if not memories:
            return "unknown"
        rendered = "\n".join(f"- {m}" for m in memories)
        today = f"Current date: {reference_time}\n" if reference_time else ""
        user = f"{today}Facts:\n{rendered}\n\nQuestion: {question}"
        return self.reader(self.reader_system, user).strip()
