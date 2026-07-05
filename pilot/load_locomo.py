"""Loader + stats for LoCoMo (locomo10.json).

Verifies the figures cited in docs/evaluation-benchmarks.md: 10
conversations, 1,986 QA pairs, sessions per conversation, tokens per
conversation, and the QA category breakdown.

Usage: python3 pilot/load_locomo.py data/locomo/locomo10.json
"""

import json
import re
import sys
from collections import Counter

CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}
# Category-id mapping varies across LoCoMo writeups; verify against the
# paper's appendix before using names in the paper. Ids themselves are
# stable in the released JSON.


def approx_tokens(text):
    return len(text) // 4


def main(path):
    data = json.load(open(path))
    print(f"file: {path}")
    print(f"conversations: {len(data)}")

    qa_total = 0
    cats = Counter()
    sess_counts = []
    tok_counts = []
    for conv in data:
        qa = conv["qa"]
        qa_total += len(qa)
        cats.update(q.get("category") for q in qa)
        c = conv["conversation"]
        sess_keys = [k for k in c if re.fullmatch(r"session_\d+", k)]
        sess_counts.append(len(sess_keys))
        toks = 0
        for k in sess_keys:
            for turn in c[k]:
                toks += approx_tokens(turn.get("text") or "")
        tok_counts.append(toks)

    print(f"QA pairs total: {qa_total}")
    print("QA per category id:")
    for cid, n in sorted(cats.items(), key=lambda x: (x[0] is None, x[0])):
        name = CATEGORY_NAMES.get(cid, "?")
        print(f"  {cid} ({name}): {n}")

    sess_counts.sort()
    tok_counts.sort()
    print(f"sessions/conversation: min {sess_counts[0]}, max {sess_counts[-1]}")
    print(f"approx tokens/conversation: min {tok_counts[0] / 1e3:.1f}K, "
          f"median {tok_counts[len(tok_counts) // 2] / 1e3:.1f}K, "
          f"max {tok_counts[-1] / 1e3:.1f}K, sum {sum(tok_counts) / 1e3:.0f}K")

    # sample QA fields
    q0 = data[0]["qa"][0]
    print(f"qa fields: {sorted(q0.keys())}")


if __name__ == "__main__":
    main(sys.argv[1])
