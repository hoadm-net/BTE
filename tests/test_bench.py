import json
import threading
import time

from bte.bench import (QARecord, render_session, run_benchmark,
                       run_longmemeval_question, summarize)


class StubMemory:
    def __init__(self, reply):
        self.reply = reply
        self.ingested = []

    def ingest_session(self, turns, timestamp):
        self.ingested.append((tuple(turns), timestamp))

    def answer(self, question, reference_time=None):
        return self.reply


class ExactJudge:
    def __call__(self, question, gold, answer):
        return gold.lower() in answer.lower()


QUESTION = {
    "question_id": "q1",
    "question_type": "knowledge-update",
    "question": "Where does the user live?",
    "answer": "Austin",
    "question_date": "2026-06-30",
    "haystack_dates": ["2026-06-01", "2026-06-02"],
    "haystack_sessions": [
        [{"role": "user", "content": "I live in Seattle."}],
        [{"role": "user", "content": "I moved to Austin."}],
    ],
}


def test_render_session_prefixes_roles():
    text = render_session(QUESTION["haystack_sessions"][0])
    assert text == "user: I live in Seattle."


def test_run_question_correct_and_timed():
    rec = run_longmemeval_question(
        lambda: StubMemory("The user lives in Austin."),
        QUESTION, ExactJudge(), "stub")
    assert rec.correct and rec.error is None
    assert rec.category == "knowledge-update"
    assert rec.approx_input_tokens > 0


def test_run_question_isolates_errors():
    class Boom(StubMemory):
        def answer(self, *a, **k):
            raise RuntimeError("api down")
    rec = run_longmemeval_question(
        lambda: Boom(""), QUESTION, ExactJudge(), "stub")
    assert not rec.correct and "api down" in rec.error


def test_run_benchmark_writes_jsonl_and_summary(tmp_path):
    out = tmp_path / "res.jsonl"
    records = run_benchmark(
        lambda: StubMemory("Austin"), [QUESTION, QUESTION],
        ExactJudge(), "stub", str(out), max_workers=2)
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["system"] == "stub"
    s = summarize(records)
    assert s["accuracy"] == 1.0
    assert s["by_category"] == {"knowledge-update": "2/2"}


def test_run_benchmark_writes_incrementally_not_at_the_end(tmp_path):
    """A crash partway through a large run must not lose the questions
    that had already finished - each record is flushed to disk as its
    own question completes, not buffered until the whole batch is done.
    Whichever of the two factory calls happens second is made to block,
    regardless of which question it ends up answering, so the test does
    not depend on the thread pool's internal scheduling order."""
    out = tmp_path / "res.jsonl"
    release_blocked = threading.Event()
    call_count = {"n": 0}
    count_lock = threading.Lock()

    class SlowMemory(StubMemory):
        def __init__(self, blocks):
            super().__init__("ok")
            self.blocks = blocks

        def answer(self, question, reference_time=None):
            if self.blocks:
                release_blocked.wait(timeout=5)
            return "ok"

    def factory():
        with count_lock:
            call_count["n"] += 1
            blocks = call_count["n"] == 2
        return SlowMemory(blocks)

    q1 = dict(QUESTION, question_id="q1")
    q2 = dict(QUESTION, question_id="q2")

    runner = threading.Thread(
        target=run_benchmark,
        args=(factory, [q1, q2], ExactJudge(), "stub", str(out)),
        kwargs={"max_workers": 2},
    )
    runner.start()

    seen = False
    for _ in range(100):
        if out.exists() and out.read_text().strip():
            seen = True
            break
        time.sleep(0.05)

    release_blocked.set()
    runner.join(timeout=5)
    assert seen, ("a finished question's record should already be on "
                 "disk while another question is still running")
    assert not runner.is_alive()


def test_summarize_empty():
    assert summarize([])["accuracy"] is None


def test_qarecord_schema_stable():
    fields = set(QARecord.__dataclass_fields__)
    assert {"system", "benchmark", "category", "question_id", "correct",
            "model_answer", "gold", "ingest_seconds", "answer_seconds",
            "approx_input_tokens", "error", "gold_in_graph",
            "ingest_errors"} == fields


def test_extract_json_variants():
    from bte.llm import CachedLLM
    f = CachedLLM._extract_json
    assert f('{"a": 1}') == {"a": 1}
    assert f('```json\n{"a": 1}\n```') == {"a": 1}
    assert f('{"facts": []}\n{"retractions": []}') == {
        "facts": [], "retractions": []}
    assert f('noise {"a": 1} trailing words') == {"a": 1}
    import pytest
    with pytest.raises(ValueError):
        f("no json here at all")
