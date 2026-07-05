"""Loader + stats for the BEAM 1M tier (data/beam-1m/).

Resolves the unconfirmed items in docs/evaluation-benchmarks.md: per-tier
conversation count (35 at 1M, from the repo README) and per-ability question
counts (from the released probing_questions.json files).

Usage: python3 pilot/load_beam.py data/beam-1m
"""

import json
import os
import sys
from collections import Counter


def approx_tokens(text):
    return len(text) // 4


def main(root):
    chat_dirs = sorted(
        (d for d in os.listdir(root) if d.isdigit()), key=int)
    print(f"root: {root}")
    print(f"chats: {len(chat_dirs)}")

    abilities = Counter()
    difficulty = Counter()
    tok_counts = []
    for d in chat_dirs:
        pq_path = os.path.join(root, d, "probing_questions.json")
        with open(pq_path) as f:
            pq = json.load(f)
        for ability, questions in pq.items():
            abilities[ability] += len(questions)
            difficulty.update(q.get("difficulty") for q in questions)
        chat_path = os.path.join(root, d, "chat.json")
        if os.path.exists(chat_path):
            with open(chat_path) as f:
                chat = json.load(f)
            if isinstance(chat, list):
                toks = sum(approx_tokens(str(t)) for t in chat)
            else:
                toks = approx_tokens(json.dumps(chat))
            tok_counts.append(toks)

    total = sum(abilities.values())
    print(f"\nprobing questions total (1M tier): {total}")
    print("per ability:")
    for a, n in abilities.most_common():
        print(f"  {a}: {n}")
    print("per difficulty:", dict(difficulty))

    if tok_counts:
        tok_counts.sort()
        n = len(tok_counts)
        print(f"\napprox tokens/chat: min {tok_counts[0] / 1e6:.2f}M, "
              f"median {tok_counts[n // 2] / 1e6:.2f}M, "
              f"max {tok_counts[-1] / 1e6:.2f}M, "
              f"sum {sum(tok_counts) / 1e6:.0f}M")


if __name__ == "__main__":
    main(sys.argv[1])
