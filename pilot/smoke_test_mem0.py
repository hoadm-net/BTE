"""1b.3 smoke test: does an off-the-shelf single-edge system (Mem0) leave
derived facts stale after a contradiction, as Theorem 1 predicts?

Three mini-conversations shaped like formalism.md's counterexample:
assert A -> user states B derived from A -> contradict A -> probe B.
A JTS-sound system either updates/invalidates B or flags the conflict;
a single-edge system answers from stale B.

Not a paper result — a reality check before building the full system.

Usage:
  OPENAI_API_KEY=... .venv/bin/python pilot/smoke_test_mem0.py
Writes a transcript to .plan/results/pilot/smoke_test_mem0_output.md
"""

import os
import shutil
import sys
import tempfile

from mem0 import Memory
from openai import OpenAI

SCENARIOS = [
    {
        "name": "correction-transaction-axis",
        "turns": [
            "I work at Acme Corp as a data analyst.",
            "Since I work at Acme, my health insurance is through Acme's provider, Aetna.",
            "Actually, I need to correct something: I never worked at Acme. I work at Apex Analytics - I mixed up the names earlier.",
        ],
        "probe": "Who provides my health insurance?",
        "stale_signals": ["aetna"],
        "expected": "Should NOT confidently answer Aetna-via-Acme; the premise was corrected.",
    },
    {
        "name": "update-valid-axis",
        "turns": [
            "I live in Seattle.",
            "My dentist is Dr. Lee, whose clinic is in Seattle - convenient since I live there.",
            "Big news: I moved to Austin last month, the relocation is complete.",
        ],
        "probe": "I need a dental cleaning next month. Where should I book it?",
        "stale_signals": ["dr. lee", "seattle"],
        "expected": "Should not recommend booking with Dr. Lee in Seattle without flagging the move.",
    },
    {
        "name": "two-hop-chain",
        "turns": [
            "I'm training for the Boston Marathon next April.",
            "Because of the marathon training, I do my long runs every Saturday morning.",
            "Since my Saturday mornings are taken by long runs, never schedule meetings for me on Saturday mornings.",
            "Update: I've withdrawn from the marathon - my knee injury won't heal in time.",
        ],
        "probe": "Can I take a recurring Saturday 9am meeting slot?",
        "stale_signals": ["no", "long run", "training"],
        "expected": "Blocking rule was derived from training, which was derived from the marathon; after withdrawal, a sound system re-opens Saturday or flags the chain.",
    },
]


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set (put it in the environment or .env)")

    client = OpenAI()
    out_lines = ["# Mem0 smoke test transcript\n"]
    for sc in SCENARIOS:
        store = tempfile.mkdtemp(prefix=f"qdrant-{sc['name']}-")
        mem = Memory.from_config({
            "vector_store": {
                "provider": "qdrant",
                "config": {"path": store, "collection_name": "smoke"},
            }
        })  # fresh, isolated store per scenario
        user_id = f"smoke-{sc['name']}"
        out_lines.append(f"\n## {sc['name']}\n")
        for t in sc["turns"]:
            mem.add(t, user_id=user_id)
            out_lines.append(f"- ingested: {t}")

        stored = mem.get_all(filters={"user_id": user_id})
        out_lines.append(f"\nstored memories after ingestion:")
        for m in stored.get("results", stored) if isinstance(stored, dict) else stored:
            out_lines.append(f"  * {m.get('memory', m)}")

        hits = mem.search(sc["probe"], filters={"user_id": user_id})
        hit_texts = [h.get("memory", str(h)) for h in
                     (hits.get("results", hits) if isinstance(hits, dict) else hits)]
        out_lines.append(f"\nprobe: {sc['probe']}")
        out_lines.append("retrieved: " + " | ".join(hit_texts))

        answer = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": "Answer using ONLY these memories about the user:\n"
                            + "\n".join(f"- {t}" for t in hit_texts)},
                {"role": "user", "content": sc["probe"]},
            ],
        ).choices[0].message.content
        out_lines.append(f"\nanswer: {answer}")
        out_lines.append(f"expected: {sc['expected']}")
        del mem
        shutil.rmtree(store, ignore_errors=True)

    path = ".plan/results/pilot/smoke_test_mem0_output.md"
    with open(path, "w") as f:
        f.write("\n".join(out_lines))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
