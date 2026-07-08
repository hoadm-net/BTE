"""LLM-free tests for the probe benchmark harness: stub memory/judge,
checks pre/post scoring, the 'unknown' gold sentinel, incremental
writes, and hop-depth breakdown - the properties the real probe run
(pilot/run_probe.py) depends on.
"""

import json

from bte.probe import generate
from bte.probe_bench import (ProbeRecord, run_probe_benchmark,
                             run_probe_item, summarize_probe)

ITEM = next(i for i in generate() if i.hop_depth == 2
           and i.contradicted == "asserted")


class StubMemory:
    def __init__(self, pre_reply, post_reply):
        self.pre_reply, self.post_reply = pre_reply, post_reply
        self.turns_seen = 0

    def ingest_session(self, turns, timestamp):
        self.turns_seen += 1

    def answer(self, question, reference_time=None):
        # first answer() call is the pre-contradiction probe
        return self.pre_reply if self.turns_seen < len(ITEM.sessions) \
            else self.post_reply


class ExactJudge:
    def __init__(self):
        self.calls = []

    DECLINE_PHRASES = ("don't know", "do not know", "unknown", "cannot")

    def __call__(self, question, gold, answer):
        self.calls.append((gold, answer))
        if gold == "unknown":
            return any(p in answer.lower() for p in self.DECLINE_PHRASES)
        return gold.lower() in answer.lower()


def factory_for(pre_reply, post_reply):
    def make(item):
        return StubMemory(pre_reply, post_reply)
    return make


def test_pre_and_post_scored_independently():
    judge = ExactJudge()
    rec = run_probe_item(
        factory_for(ITEM.gold_pre, "I don't know"), ITEM, judge, "stub")
    assert rec.pre_correct
    # gold_post for an 'asserted' contradiction at depth>=2 is "unknown"
    assert ITEM.gold_post == "unknown"
    assert rec.post_correct
    assert rec.hop_depth == 2 and rec.axis == ITEM.axis
    assert rec.domain == ITEM.domain


def test_confident_wrong_guess_fails_unknown_gold():
    judge = ExactJudge()
    rec = run_probe_item(
        factory_for(ITEM.gold_pre, "It is definitely still the old value"),
        ITEM, judge, "stub")
    assert rec.pre_correct
    assert not rec.post_correct


def test_error_isolated_per_item():
    class Boom:
        def ingest_session(self, *a, **k):
            raise RuntimeError("network down")

    def factory(item):
        return Boom()

    rec = run_probe_item(factory, ITEM, ExactJudge(), "stub")
    assert not rec.pre_correct and not rec.post_correct
    assert "network down" in rec.error


def test_run_probe_benchmark_writes_and_summarizes(tmp_path):
    out = tmp_path / "probe.jsonl"
    items = [i for i in generate() if i.hop_depth in (1, 3)][:4]
    records = run_probe_benchmark(
        factory_for("x", "x"), items, ExactJudge(), "stub", str(out),
        max_workers=2)
    lines = out.read_text().strip().splitlines()
    assert len(lines) == len(items)
    assert json.loads(lines[0])["system"] == "stub"
    s = summarize_probe(records)
    assert s["n"] == len(items)
    assert set(s["post_by_hop_depth"]) <= {1, 3}


def test_summarize_probe_empty():
    assert summarize_probe([])["post_accuracy"] is None


def test_record_schema_stable():
    fields = set(ProbeRecord.__dataclass_fields__)
    assert {"system", "probe_id", "domain", "hop_depth", "contradicted",
            "axis", "density", "confidence", "gold_pre", "gold_post",
            "pre_answer", "post_answer", "pre_correct", "post_correct",
            "ingest_seconds", "error"} == fields
