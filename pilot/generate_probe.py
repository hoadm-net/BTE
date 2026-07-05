"""Generate the diagnostic probe set (versioned, deterministic seed).

Usage: uv run python pilot/generate_probe.py
Writes data/probe/probe_v0.json and prints design stats + checksum.
The artifact is reproducible from the generator + seed; the freeze
(checksummed, committed) happens after surface paraphrase and human
validation in Phase 4 proper.
"""

import hashlib
import os
from collections import Counter

from bte.probe import generate, dump

OUT = "data/probe/probe_v0.json"


def main():
    items = generate()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    dump(items, OUT)
    digest = hashlib.sha256(open(OUT, "rb").read()).hexdigest()

    print(f"items: {len(items)}")
    for axis_name in ("hop_depth", "contradicted", "axis", "density",
                      "confidence", "domain"):
        c = Counter(getattr(i, axis_name) for i in items)
        print(f"  {axis_name}: {dict(sorted(c.items(), key=str))}")
    turns = sum(len(s) for i in items for s in i.sessions)
    print(f"total user turns: {turns}")
    print(f"sha256: {digest}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
