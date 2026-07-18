"""Small utilities shared across cache-backed clients."""

from __future__ import annotations

import os
import threading
from pathlib import Path


def atomic_write_text(path: Path, content: str) -> None:
    """Write via a temp file + os.replace so concurrent writers racing on
    the same cache key never interleave into a corrupted file. Plain
    write_text() is not atomic across threads/processes: two
    ThreadPoolExecutor workers both missing the same cache entry can
    both open the path for writing and their writes can interleave,
    producing a file with two concatenated JSON blobs (observed: 2 of
    33k+ embedding cache entries corrupted exactly this way,
    deterministically breaking every probe item whose retrieval needed
    that entry - not a network/provider issue, a write-write race). The
    temp filename includes both pid and thread id since same-process
    threads share a pid. os.replace is atomic on POSIX and Windows, so
    the final file is always one writer's complete output.
    """
    tmp = path.with_suffix(f".{os.getpid()}-{threading.get_ident()}.tmp")
    tmp.write_text(content)
    os.replace(tmp, path)
