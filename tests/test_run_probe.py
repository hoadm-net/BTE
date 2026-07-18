"""already_done() for the probe runner. probe_bench.py's ProbeRecord has
no per-session error swallowing (see bte/probe_bench.py's run_probe_item
docstring), so error=None alone is a safe completion signal here -
unlike run_longmemeval.py's version, which needed an extra ingest_errors
threshold after the mem0 100-question run's quiet-corruption bug. This
test locks in that error=None stays sufficient; if probe_bench.py ever
grows per-session isolation, this test (and already_done) must be
revisited together."""

import json

from pilot.run_probe import already_done


def write(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_clean_success_counts_as_done(tmp_path):
    out = tmp_path / "r.jsonl"
    write(out, [{"system": "mem0", "probe_id": "p1", "error": None}])
    assert already_done(str(out), "mem0") == {"p1"}


def test_error_not_done(tmp_path):
    out = tmp_path / "r.jsonl"
    write(out, [{"system": "mem0", "probe_id": "p1", "error": "boom"}])
    assert already_done(str(out), "mem0") == set()


def test_latest_record_wins(tmp_path):
    out = tmp_path / "r.jsonl"
    write(out, [
        {"system": "mem0", "probe_id": "p1", "error": "boom"},
        {"system": "mem0", "probe_id": "p1", "error": None},
    ])
    assert already_done(str(out), "mem0") == {"p1"}


def test_latest_failure_after_earlier_success_is_not_done(tmp_path):
    """The actual bug: an item succeeded under an older prompt/schema,
    then failed on the latest attempt after a code change (observed:
    33 probe items succeeded before an enum-constrained relation_vocab
    change, then failed under it) - naive 'ever succeeded' resume would
    keep serving the stale pre-change success forever and never retry."""
    out = tmp_path / "r.jsonl"
    write(out, [
        {"system": "bjg", "probe_id": "p1", "error": None},
        {"system": "bjg", "probe_id": "p1", "error": None},
        {"system": "bjg", "probe_id": "p1", "error": "JSONDecodeError: boom"},
    ])
    assert already_done(str(out), "bjg") == set()


def test_scoped_per_system(tmp_path):
    out = tmp_path / "r.jsonl"
    write(out, [{"system": "bjg", "probe_id": "p1", "error": None}])
    assert already_done(str(out), "mem0") == set()


def test_missing_file_returns_empty(tmp_path):
    assert already_done(str(tmp_path / "nope.jsonl"), "bjg") == set()
