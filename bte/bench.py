"""Benchmark harness: dataset loading, the shared results schema, and a
runner that drives any MemorySystem (.plan 03: one results table that
every later analysis reads; fixed reader and judge across systems).

Token counts are chars/4 estimates so they are reproducible from cache;
exact billed usage is provider-side accounting, not part of the schema.
Wall-clock times are recorded but API jitter makes tokens the primary
cost metric (H4).
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Callable, Optional

from .llm import CachedLLM
from .memory import MemorySystem

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "correct": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["correct", "reason"],
    "additionalProperties": False,
}

JUDGE_SYSTEM = (
    "You grade a memory-augmented assistant's answer against the gold "
    "answer. The answer is correct if it conveys the gold answer's "
    "content (paraphrase allowed) without asserting anything that "
    "contradicts it. If the gold answer indicates the question is "
    "unanswerable, the response is correct only if it declines to answer."
)


@dataclass
class QARecord:
    system: str
    benchmark: str
    category: str
    question_id: str
    correct: bool
    model_answer: str
    gold: str
    ingest_seconds: float
    answer_seconds: float
    approx_input_tokens: int
    error: Optional[str] = None


def load_longmemeval(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def render_session(session: list[dict]) -> str:
    return "\n".join(
        f"{t.get('role', 'user')}: {t.get('content') or ''}" for t in session)


class Judge:
    def __init__(self, llm: CachedLLM) -> None:
        self.llm = llm

    def __call__(self, question: str, gold: str, answer: str) -> bool:
        verdict = self.llm.complete_json(
            JUDGE_SYSTEM,
            f"Question: {question}\nGold answer: {gold}\n"
            f"Assistant answer: {answer}",
            "judgment", JUDGE_SCHEMA,
        )
        return bool(verdict["correct"])


def run_longmemeval_question(
    memory_factory: Callable[[], MemorySystem],
    q: dict,
    judge: Judge,
    system: str,
) -> QARecord:
    memory = memory_factory()
    sessions = q["haystack_sessions"]
    dates = q.get("haystack_dates") or [""] * len(sessions)
    approx_tokens = 0
    t0 = time.monotonic()
    error = None
    try:
        for session, date in zip(sessions, dates):
            text = render_session(session)
            approx_tokens += len(text) // 4
            memory.ingest_session([text], date or "unknown")
        t1 = time.monotonic()
        answer = memory.answer(q["question"],
                               reference_time=q.get("question_date"))
        t2 = time.monotonic()
        correct = judge(q["question"], str(q["answer"]), answer)
    except Exception as exc:
        t1 = t2 = time.monotonic()
        answer, correct = "", False
        error = f"{type(exc).__name__}: {exc}"
    return QARecord(
        system=system, benchmark="longmemeval_s",
        category=q.get("question_type", "?"),
        question_id=str(q.get("question_id")),
        correct=correct, model_answer=answer, gold=str(q["answer"]),
        ingest_seconds=round(t1 - t0, 2),
        answer_seconds=round(t2 - t1, 2),
        approx_input_tokens=approx_tokens,
        error=error,
    )


def run_benchmark(
    memory_factory: Callable[[], MemorySystem],
    questions: list[dict],
    judge: Judge,
    system: str,
    out_path: str,
    max_workers: int = 4,
) -> list[QARecord]:
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        records = list(pool.map(
            lambda q: run_longmemeval_question(
                memory_factory, q, judge, system),
            questions))
    with open(out_path, "a") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")
    return records


def summarize(records: list[QARecord]) -> dict:
    by_cat: dict[str, list[QARecord]] = {}
    for r in records:
        by_cat.setdefault(r.category, []).append(r)
    return {
        "n": len(records),
        "accuracy": round(
            sum(r.correct for r in records) / len(records), 3)
        if records else None,
        "errors": sum(1 for r in records if r.error),
        "by_category": {
            c: f"{sum(r.correct for r in rs)}/{len(rs)}"
            for c, rs in sorted(by_cat.items())
        },
    }
