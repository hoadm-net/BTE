"""already_done() resume logic: a record must be error-free AND have a
low ingest_errors count to count as done - a question can finish with
no top-level error while most of its sessions were silently dropped
(the mem0 100-question run: an OpenRouter outage left ~30 'successful'
records with 40+ of ~48 sessions skipped, all answering 'unknown' from
a near-empty memory)."""

import json

from pilot.run_longmemeval import already_done


def write(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_clean_success_counts_as_done(tmp_path):
    out = tmp_path / "r.jsonl"
    write(out, [{"system": "mem0", "question_id": "q1", "error": None,
                "ingest_errors": 0}])
    assert already_done(str(out), "mem0") == {"q1"}


def test_top_level_error_not_done(tmp_path):
    out = tmp_path / "r.jsonl"
    write(out, [{"system": "mem0", "question_id": "q1",
                "error": "boom", "ingest_errors": 0}])
    assert already_done(str(out), "mem0") == set()


def test_quietly_corrupted_success_not_done(tmp_path):
    """error=None but 44 of ~48 sessions silently dropped - this must
    NOT be treated as done."""
    out = tmp_path / "r.jsonl"
    write(out, [{"system": "mem0", "question_id": "q1", "error": None,
                "ingest_errors": 44}])
    assert already_done(str(out), "mem0") == set()


def test_few_transient_skips_still_counts_as_done(tmp_path):
    out = tmp_path / "r.jsonl"
    write(out, [{"system": "mem0", "question_id": "q1", "error": None,
                "ingest_errors": 2}])
    assert already_done(str(out), "mem0") == {"q1"}


def test_latest_record_wins_on_reprocessing(tmp_path):
    out = tmp_path / "r.jsonl"
    write(out, [
        {"system": "mem0", "question_id": "q1", "error": None,
         "ingest_errors": 44},
        {"system": "mem0", "question_id": "q1", "error": None,
         "ingest_errors": 0},
    ])
    assert already_done(str(out), "mem0") == {"q1"}


def test_scoped_per_system(tmp_path):
    out = tmp_path / "r.jsonl"
    write(out, [{"system": "bjg", "question_id": "q1", "error": None,
                "ingest_errors": 0}])
    assert already_done(str(out), "mem0") == set()
