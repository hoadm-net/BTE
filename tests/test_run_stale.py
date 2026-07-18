"""already_done() for the STALE runner - must be latest-wins, mirroring
run_probe.py/run_longmemeval.py's fix (an item can succeed under an
older prompt/code path and fail under a newer one; only the most recent
attempt should count)."""

import json

from pilot.run_stale import already_done


def write(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_clean_success_counts_as_done(tmp_path):
    out = tmp_path / "r.jsonl"
    write(out, [{"system": "bjg", "uid": "u1", "error": None}])
    assert already_done(str(out), "bjg") == {"u1"}


def test_error_not_done(tmp_path):
    out = tmp_path / "r.jsonl"
    write(out, [{"system": "bjg", "uid": "u1", "error": "boom"}])
    assert already_done(str(out), "bjg") == set()


def test_latest_record_wins(tmp_path):
    out = tmp_path / "r.jsonl"
    write(out, [
        {"system": "bjg", "uid": "u1", "error": "boom"},
        {"system": "bjg", "uid": "u1", "error": None},
    ])
    assert already_done(str(out), "bjg") == {"u1"}


def test_latest_failure_after_earlier_success_is_not_done(tmp_path):
    out = tmp_path / "r.jsonl"
    write(out, [
        {"system": "bjg", "uid": "u1", "error": None},
        {"system": "bjg", "uid": "u1", "error": "boom"},
    ])
    assert already_done(str(out), "bjg") == set()


def test_scoped_per_system(tmp_path):
    out = tmp_path / "r.jsonl"
    write(out, [{"system": "mem0", "uid": "u1", "error": None}])
    assert already_done(str(out), "bjg") == set()


def test_missing_file_returns_empty(tmp_path):
    assert already_done(str(tmp_path / "nope.jsonl"), "bjg") == set()
