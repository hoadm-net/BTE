"""atomic_write_text: the write-write race that corrupted 2 of 33k+
embedding cache entries in production (two ThreadPoolExecutor workers
racing on the same cache key, plain write_text() interleaving into a
'valid JSON + extra data' file) - see bte/_util.py's docstring."""

import threading

from bte._util import atomic_write_text


def test_writes_readable_content(tmp_path):
    p = tmp_path / "x.json"
    atomic_write_text(p, "hello")
    assert p.read_text() == "hello"


def test_overwrites_existing_file(tmp_path):
    p = tmp_path / "x.json"
    atomic_write_text(p, "first")
    atomic_write_text(p, "second")
    assert p.read_text() == "second"


def test_no_leftover_tmp_files(tmp_path):
    p = tmp_path / "x.json"
    atomic_write_text(p, "content")
    leftovers = [f for f in tmp_path.iterdir() if f != p]
    assert leftovers == []


def test_concurrent_writers_never_interleave(tmp_path):
    """The actual bug reproduced: many threads racing to write the SAME
    path with DIFFERENT-length payloads. Plain write_text() under this
    load reliably produces a file containing bytes from more than one
    writer; atomic_write_text() must never do that - the final file is
    always exactly one writer's untouched content."""
    p = tmp_path / "race.json"
    payloads = [f'{{"n": {i}, "pad": "{"x" * i}"}}' for i in range(50)]

    def write(payload):
        atomic_write_text(p, payload)

    threads = [threading.Thread(target=write, args=(pl,)) for pl in payloads]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    final = p.read_text()
    assert final in payloads, "final content must be exactly one writer's payload"
