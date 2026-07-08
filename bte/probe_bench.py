"""Diagnostic probe benchmark harness (.plan 04-probe.md, 05-experiments.md
E1/E2): runs the probe's canned conversation surface through real
extraction/ingestion and scores pre- and post-contradiction answers
against gold. This is the instrument for H1 (baseline accuracy drops at
hop >= 2) and H2 (ours does not) - unlike LongMemEval's KU/TR tiers,
hop depth is a controlled design axis here, not an incidental property
of the data.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Callable, Optional

from .llm import CachedLLM
from .memory import MemorySystem
from .probe import ProbeItem

PROBE_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "correct": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["correct", "reason"],
    "additionalProperties": False,
}

PROBE_JUDGE_SYSTEM = (
    "You grade a memory-augmented assistant's answer against the gold "
    "answer. The answer is correct if it conveys the gold answer's "
    "content (paraphrase allowed) without asserting anything that "
    "contradicts it. The special gold value 'unknown' means the correct "
    "fact is no longer determinable (the premise was contradicted or "
    "withdrawn with no stated replacement) - the response is correct "
    "only if it declines to give a specific answer (says it does not "
    "know, cannot determine this, or the information is no longer "
    "available). A confident specific answer when gold is 'unknown' is "
    "always wrong, even if the guessed value happens to match a prior "
    "state."
)


class ProbeJudge:
    def __init__(self, llm: CachedLLM) -> None:
        self.llm = llm

    def __call__(self, question: str, gold: str, answer: str) -> bool:
        verdict = self.llm.complete_json(
            PROBE_JUDGE_SYSTEM,
            f"Question: {question}\nGold answer: {gold}\n"
            f"Assistant answer: {answer}",
            "judgment", PROBE_JUDGE_SCHEMA,
        )
        return bool(verdict["correct"])


@dataclass
class ProbeRecord:
    system: str
    probe_id: str
    domain: str
    hop_depth: int
    contradicted: str
    axis: str
    density: str
    confidence: str
    gold_pre: str
    gold_post: str
    pre_answer: str
    post_answer: str
    pre_correct: bool
    post_correct: bool
    ingest_seconds: float
    error: Optional[str] = None


def run_probe_item(
    memory_factory: Callable[[ProbeItem], MemorySystem],
    item: ProbeItem,
    judge: ProbeJudge,
    system: str,
) -> ProbeRecord:
    """Deliberately all-or-nothing, unlike bench.py's per-session
    isolation: a probe item has few sessions (<=9) and every one is a
    designed fact, not filler - silently dropping the contradiction turn
    itself would corrupt exactly what the item is measuring. bench.py's
    per-session catch is right for LongMemEval's 48-session haystacks
    (cost of retrying one long history is high, and losing one filler
    session is harmless); it is the wrong tradeoff here. A run that
    fails partway must raise, not return a record that looks clean -
    the mem0 100-question LongMemEval run silently absorbed an
    OpenRouter outage into 'successful' records this way."""
    memory = memory_factory(item)
    n = len(item.sessions)
    t0 = time.monotonic()
    pre_answer = post_answer = ""
    pre_correct = post_correct = False
    error = None
    try:
        for i in range(n - 1):
            text = " ".join(item.sessions[i])
            memory.ingest_session([text], f"2026-06-{i + 1:02d}")
        pre_answer = memory.answer(
            item.question, reference_time=f"2026-06-{n:02d}")
        pre_correct = judge(item.question, item.gold_pre, pre_answer)

        text = " ".join(item.sessions[n - 1])
        memory.ingest_session([text], f"2026-06-{n:02d}")
        post_answer = memory.answer(
            item.question, reference_time=f"2026-06-{n + 1:02d}")
        post_correct = judge(item.question, item.gold_post, post_answer)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    t1 = time.monotonic()
    return ProbeRecord(
        system=system, probe_id=item.probe_id, domain=item.domain,
        hop_depth=item.hop_depth, contradicted=item.contradicted,
        axis=item.axis, density=item.density, confidence=item.confidence,
        gold_pre=item.gold_pre, gold_post=item.gold_post,
        pre_answer=pre_answer, post_answer=post_answer,
        pre_correct=pre_correct, post_correct=post_correct,
        ingest_seconds=round(t1 - t0, 2), error=error,
    )


def run_probe_benchmark(
    memory_factory: Callable[[ProbeItem], MemorySystem],
    items: list[ProbeItem],
    judge: ProbeJudge,
    system: str,
    out_path: str,
    max_workers: int = 4,
) -> list[ProbeRecord]:
    """Writes each record as its own item finishes (see bench.py's
    run_benchmark for why: a killed process must not lose completed
    items)."""
    records: list[ProbeRecord] = []
    lock = threading.Lock()

    def run_and_write(item: ProbeItem) -> ProbeRecord:
        r = run_probe_item(memory_factory, item, judge, system)
        with lock:
            with open(out_path, "a") as f:
                f.write(json.dumps(asdict(r)) + "\n")
        return r

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for r in pool.map(run_and_write, items):
            records.append(r)
    return records


def summarize_probe(records: list[ProbeRecord]) -> dict:
    if not records:
        return {"n": 0, "post_accuracy": None}
    by_depth: dict[int, list[int]] = {}
    for r in records:
        by_depth.setdefault(r.hop_depth, [0, 0])
        by_depth[r.hop_depth][1] += 1
        by_depth[r.hop_depth][0] += r.post_correct
    return {
        "n": len(records),
        "pre_accuracy": round(
            sum(r.pre_correct for r in records) / len(records), 3),
        "post_accuracy": round(
            sum(r.post_correct for r in records) / len(records), 3),
        "post_by_hop_depth": {
            d: f"{ok}/{tot}" for d, (ok, tot) in sorted(by_depth.items())
        },
        "errors": sum(1 for r in records if r.error),
    }
