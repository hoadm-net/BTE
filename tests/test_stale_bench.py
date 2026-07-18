"""LLM-free tests for the STALE benchmark harness: stub memory/judge,
checks per-dimension scoring, incremental writes, and that a mid-session
ingest failure is isolated (counted, not fatal) while the item still
proceeds to answer the probing queries."""

import json

from bte.stale_bench import (StaleRecord, run_stale_benchmark,
                             run_stale_item, summarize_stale)

SCENARIO = {
    "uid": "abc12345-0000-0000-0000-000000000000",
    "type": "T2",
    "M_old": "I live in Portland.",
    "M_new": "I found a bark scorpion near my boots this morning.",
    "explanation": "Bark scorpions and dry heat imply the desert "
                   "Southwest, not rainy Portland.",
    "probing_queries": {
        "dim1_query": "Does the user still live in Portland?",
        "dim2_query": "Since the user lives in Portland, suggest rainy-day activities.",
        "dim3_query": "What should the user keep on hand at home?",
    },
    "relevant_session_index": [0, 1],
    "timestamps": ["2022-01-01", "2022-02-01"],
    "haystack_session": [
        [{"role": "user", "content": "I live in Portland."}],
        [{"role": "user", "content": "Found a bark scorpion by my boots."}],
    ],
}


class StubMemory:
    def __init__(self, reply_map=None, fail_on_session=None):
        self.reply_map = reply_map or {}
        self.fail_on_session = fail_on_session
        self.sessions_seen = 0

    def ingest_session(self, turns, timestamp):
        self.sessions_seen += 1
        if self.fail_on_session == self.sessions_seen:
            raise RuntimeError("provider hiccup")

    def answer(self, question, reference_time=None):
        return self.reply_map.get(question, "unknown")


class KeywordJudge:
    """Correct iff the dimension's expected keyword appears (or is
    absent, for dim2's rejection case) in the response."""

    EXPECT_ABSENT = {"premise_resistance"}

    def __call__(self, dimension, query, response, m_old, m_new, explanation):
        has_scorpion = "scorpion" in response.lower() or "desert" in response.lower()
        if dimension in self.EXPECT_ABSENT:
            return "rainy" not in response.lower()
        return has_scorpion


def factory_for(reply_map, fail_on_session=None):
    def make(scenario):
        return StubMemory(reply_map, fail_on_session)
    return make


def test_all_three_dimensions_scored_independently():
    replies = {
        SCENARIO["probing_queries"]["dim1_query"]: "not sure, maybe desert now",
        SCENARIO["probing_queries"]["dim2_query"]: "I can't assume that's still true",
        SCENARIO["probing_queries"]["dim3_query"]: "keep scorpion repellent handy",
    }
    rec = run_stale_item(factory_for(replies), SCENARIO, KeywordJudge(), "stub")
    assert rec.dim1_correct
    assert rec.dim2_correct
    assert rec.dim3_correct
    assert rec.uid == SCENARIO["uid"]
    assert rec.error is None


def test_stale_answer_fails_all_dimensions():
    replies = {
        SCENARIO["probing_queries"]["dim1_query"]: "yes, still in Portland",
        SCENARIO["probing_queries"]["dim2_query"]: "here are rainy day ideas",
        SCENARIO["probing_queries"]["dim3_query"]: "nothing special needed",
    }
    rec = run_stale_item(factory_for(replies), SCENARIO, KeywordJudge(), "stub")
    assert not rec.dim1_correct
    assert not rec.dim2_correct
    assert not rec.dim3_correct


def test_mid_session_ingest_failure_is_isolated_not_fatal():
    rec = run_stale_item(
        factory_for({}, fail_on_session=1), SCENARIO, KeywordJudge(), "stub")
    assert rec.error is None
    assert rec.ingest_errors == 1
    # still proceeded to answer/judge despite the dropped session
    assert rec.dim1_answer == "unknown"


def test_error_before_ingestion_is_fatal():
    def factory(scenario):
        raise RuntimeError("factory exploded")
    rec = run_stale_item(factory, SCENARIO, KeywordJudge(), "stub")
    assert rec.error is not None and "factory exploded" in rec.error


def test_run_stale_benchmark_writes_and_summarizes(tmp_path):
    out = tmp_path / "stale.jsonl"
    replies = {
        SCENARIO["probing_queries"]["dim1_query"]: "not sure, desert now",
        SCENARIO["probing_queries"]["dim2_query"]: "can't assume that",
        SCENARIO["probing_queries"]["dim3_query"]: "keep scorpion spray handy",
    }
    records = run_stale_benchmark(
        factory_for(replies), [SCENARIO, SCENARIO], KeywordJudge(),
        "stub", str(out), max_workers=2)
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["system"] == "stub"
    s = summarize_stale(records)
    assert s["n"] == 2
    assert s["dim1_state_resolution"] == 1.0
    assert s["all_three_correct"] == 1.0


def test_summarize_empty():
    assert summarize_stale([]) == {"n": 0}


def test_record_schema_stable():
    fields = set(StaleRecord.__dataclass_fields__)
    assert {"system", "uid", "dim1_correct", "dim2_correct", "dim3_correct",
            "dim1_answer", "dim2_answer", "dim3_answer", "ingest_seconds",
            "ingest_errors", "error"} == fields
