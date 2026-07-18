"""STALE T2 benchmark harness (evaluation-benchmarks.md; related-work.md's
CUPMem entry). External, expert-validated evidence for H1/H2: unlike the
diagnostic probe, STALE T2's ~50-session haystacks don't fit inside a
retrieval window, so a system cannot "cheat" by having the reader
re-derive the update from raw context the way it can on the probe.

Follows STALE's own evaluation protocol (checked 2026-07-08): judge
against the scenario's own old/new-state description ("foundational
state logic"), not a synthetic reference string, scoring three
dimensions per item - State Resolution (dim1), Premise Resistance
(dim2), Implicit Policy Adaptation (dim3).
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Callable, Optional

from .bench import render_session
from .llm import CachedLLM
from .memory import MemorySystem

# Aligned with STALE's own response-generation prompts (arXiv 2605.06527
# Appendix E.1: "You are a helpful assistant. Review the following
# conversation history with the user, then accurately answer the
# question." - no abstention clause). Ours reviews rendered memory facts
# instead of raw history, so the rendering annotations are explained,
# but there is deliberately no "reply unknown" instruction.
STALE_READER_SYSTEM = (
    "You are a helpful assistant with access to facts recorded from "
    "earlier conversations with the user. Review the facts, then "
    "respond to the user's question directly and accurately.\n"
    "Each fact ends with '(recorded DATE)' - the date the user said "
    "it. A fact may carry change history in parentheses: '(previously: "
    "X, recorded DATE)' means its value was X before being updated. "
    "When facts disagree, the more recently recorded one reflects the "
    "user's current situation."
)

STALE_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "correct": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["correct", "reason"],
    "additionalProperties": False,
}

STALE_JUDGE_SYSTEM = """You evaluate whether an assistant's response demonstrates correct awareness of a user's state change. Judge against the underlying situation described below, not against any fixed reference wording - the response can use entirely different words and still be correct.

Situation:
- OLD state (established earlier, no longer current): {m_old}
- NEW state (implied by a later message, never stated explicitly by the user): {m_new}
- Why the new state invalidates the old one: {explanation}

You are judging the "{dimension}" dimension. {instruction}

Question asked: {query}
Assistant's response: {response}

Decide pass/fail per the dimension's criterion above."""

DIMENSION_INSTRUCTIONS = {
    "state_resolution": (
        "This question asks directly whether the OLD state still holds. "
        "PASS only if the response indicates the old state is no longer "
        "certain or true - e.g. expresses doubt, declines to confirm it, "
        "or states the current/new situation instead. FAIL if it "
        "confidently affirms the old state as still current."
    ),
    "premise_resistance": (
        "This question presupposes the OLD state is still true and asks "
        "for help based on it, without stating the new state. PASS only "
        "if the response does NOT simply comply with the false premise - "
        "e.g. it questions or flags the premise, asks for clarification, "
        "or notes a possible inconsistency, rather than helpfully "
        "answering as if the premise holds. FAIL if it proceeds to "
        "answer as though the old state were still true."
    ),
    "implicit_policy_adaptation": (
        "This question does not mention the old or new state explicitly, "
        "and requires using the CURRENT (new) state to give an "
        "appropriate answer. PASS only if the response's content is "
        "consistent with the new state's implications, not the old "
        "state's. FAIL if the response assumes or reflects the old state."
    ),
}

DIMENSION_KEYS = (
    ("state_resolution", "dim1_query"),
    ("premise_resistance", "dim2_query"),
    ("implicit_policy_adaptation", "dim3_query"),
)


class StaleJudge:
    def __init__(self, llm: CachedLLM) -> None:
        self.llm = llm

    def __call__(self, dimension: str, query: str, response: str,
                m_old: str, m_new: str, explanation: str) -> bool:
        system = STALE_JUDGE_SYSTEM.format(
            m_old=m_old, m_new=m_new, explanation=explanation,
            dimension=dimension,
            instruction=DIMENSION_INSTRUCTIONS[dimension],
            query=query, response=response,
        )
        verdict = self.llm.complete_json(
            system, "Respond with your pass/fail judgment.",
            "judgment", STALE_JUDGE_SCHEMA,
        )
        return bool(verdict["correct"])


@dataclass
class StaleRecord:
    system: str
    uid: str
    dim1_correct: bool
    dim2_correct: bool
    dim3_correct: bool
    dim1_answer: str
    dim2_answer: str
    dim3_answer: str
    ingest_seconds: float
    ingest_errors: int = 0
    error: Optional[str] = None


def run_stale_item(
    memory_factory: Callable[[dict], MemorySystem],
    scenario: dict,
    judge: StaleJudge,
    system: str,
) -> StaleRecord:
    t0 = time.monotonic()
    answers = {"dim1_query": "", "dim2_query": "", "dim3_query": ""}
    correct = {"state_resolution": False, "premise_resistance": False,
              "implicit_policy_adaptation": False}
    ingest_errors = 0
    error = None
    try:
        memory = memory_factory(scenario)
        sessions = scenario["haystack_session"]
        timestamps = scenario["timestamps"]
        for session, ts in zip(sessions, timestamps):
            text = render_session(session)
            try:
                memory.ingest_session([text], ts)
            except Exception:
                ingest_errors += 1
        ref_time = timestamps[-1]
        for dim, qkey in DIMENSION_KEYS:
            query = scenario["probing_queries"][qkey]
            answer = memory.answer(query, reference_time=ref_time)
            answers[qkey] = answer
            correct[dim] = judge(
                dim, query, answer, scenario["M_old"], scenario["M_new"],
                scenario["explanation"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    t1 = time.monotonic()
    return StaleRecord(
        system=system, uid=scenario["uid"],
        dim1_correct=correct["state_resolution"],
        dim2_correct=correct["premise_resistance"],
        dim3_correct=correct["implicit_policy_adaptation"],
        dim1_answer=answers["dim1_query"], dim2_answer=answers["dim2_query"],
        dim3_answer=answers["dim3_query"],
        ingest_seconds=round(t1 - t0, 2), ingest_errors=ingest_errors,
        error=error,
    )


def run_stale_benchmark(
    memory_factory: Callable[[dict], MemorySystem],
    scenarios: list[dict],
    judge: StaleJudge,
    system: str,
    out_path: str,
    max_workers: int = 4,
) -> list[StaleRecord]:
    records: list[StaleRecord] = []
    lock = threading.Lock()

    def run_and_write(scenario: dict) -> StaleRecord:
        r = run_stale_item(memory_factory, scenario, judge, system)
        with lock:
            with open(out_path, "a") as f:
                f.write(json.dumps(asdict(r)) + "\n")
        return r

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for r in pool.map(run_and_write, scenarios):
            records.append(r)
    return records


def summarize_stale(records: list[StaleRecord]) -> dict:
    if not records:
        return {"n": 0}
    n = len(records)
    return {
        "n": n,
        "dim1_state_resolution": round(
            sum(r.dim1_correct for r in records) / n, 3),
        "dim2_premise_resistance": round(
            sum(r.dim2_correct for r in records) / n, 3),
        "dim3_implicit_policy_adaptation": round(
            sum(r.dim3_correct for r in records) / n, 3),
        "all_three_correct": round(
            sum(r.dim1_correct and r.dim2_correct and r.dim3_correct
               for r in records) / n, 3),
        "errors": sum(1 for r in records if r.error),
    }
