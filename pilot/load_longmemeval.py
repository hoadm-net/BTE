"""Loader + stats for LongMemEval (cleaned release).

Answers the unconfirmed items flagged in docs/evaluation-benchmarks.md:
  - exact per-category question counts (KU / TR / ...)
  - whether haystack sessions are shared across questions or per-question
  - haystack size distribution (approx tokens)

Usage: python3 pilot/load_longmemeval.py data/longmemeval/longmemeval_s_cleaned.json
"""

import json
import sys
from collections import Counter


def approx_tokens(text):
    # chars/4 heuristic; fine for budgeting, not billing
    return len(text) // 4


def session_text(session):
    return " ".join(turn.get("content") or "" for turn in session)


def main(path):
    with open(path) as f:
        data = json.load(f)

    print(f"file: {path}")
    print(f"questions: {len(data)}")
    print(f"fields: {sorted(data[0].keys())}")

    types = Counter(q["question_type"] for q in data)
    print("\nper-category counts:")
    for t, n in types.most_common():
        print(f"  {t}: {n}")

    abst = sum(1 for q in data if str(q["question_id"]).endswith("_abs"))
    print(f"abstention (_abs) questions: {abst}")

    # haystack sharing: same session id appearing under multiple questions
    sess_owner = Counter()
    per_q_sessions = []
    per_q_tokens = []
    for q in data:
        ids = q.get("haystack_session_ids", [])
        per_q_sessions.append(len(ids))
        sess_owner.update(set(ids))
        toks = sum(approx_tokens(session_text(s)) for s in q.get("haystack_sessions", []))
        per_q_tokens.append(toks)

    uniq = len(sess_owner)
    total = sum(per_q_sessions)
    shared = sum(1 for c in sess_owner.values() if c > 1)
    print(f"\nhaystack sessions: total refs {total}, unique ids {uniq}, "
          f"ids appearing in >1 question: {shared} ({100 * shared / max(uniq, 1):.1f}%)")

    per_q_sessions.sort()
    per_q_tokens.sort()
    n = len(data)
    print(f"sessions/question: min {per_q_sessions[0]}, median {per_q_sessions[n // 2]}, "
          f"max {per_q_sessions[-1]}")
    print(f"approx tokens/haystack: min {per_q_tokens[0] / 1e3:.0f}K, "
          f"median {per_q_tokens[n // 2] / 1e3:.0f}K, max {per_q_tokens[-1] / 1e3:.0f}K, "
          f"sum {sum(per_q_tokens) / 1e6:.1f}M")

    # evidence chain length for the two categories this project targets
    for cat in ("knowledge-update", "temporal-reasoning"):
        ev = sorted(len(q.get("answer_session_ids", [])) for q in data
                    if q["question_type"] == cat)
        if ev:
            print(f"evidence sessions/question [{cat}]: min {ev[0]}, "
                  f"median {ev[len(ev) // 2]}, max {ev[-1]}")


if __name__ == "__main__":
    main(sys.argv[1])
