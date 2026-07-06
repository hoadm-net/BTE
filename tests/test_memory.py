"""LLM-free end-to-end memory tests: structured facts in, reader prompt
out — checks that what BBP supersedes disappears from the reader's view
(the property Mem0's smoke test failed on).
"""

from bte.ingest import Ingestor
from bte.memory import BJGMemory
from bte.retrieval import Retriever
from bte.rules import ChainRule


def fact(subject, relation, object, correction=False):
    return {"subject": subject, "relation": relation, "object": object,
            "valid_from": None, "valid_to": None, "confidence": 1.0,
            "is_correction": correction, "premises": []}


class PromptCapture:
    def __init__(self):
        self.prompts = []

    def __call__(self, system, user):
        self.prompts.append(user)
        return "stub"


def build_memory():
    ing = Ingestor(rules=[ChainRule("works_at", "insurer_of", "insured_by")])
    reader = PromptCapture()
    mem = BJGMemory(ing, Retriever(ing.graph), reader)
    return mem, reader


def test_answer_reflects_propagated_state():
    mem, reader = build_memory()
    mem.ingest_structured([
        fact("user", "works_at", "acme"),
        fact("acme", "insurer_of", "Aetna"),
    ], [], "2026-06-01")

    mem.answer("who insures the user?")
    assert "user | insured_by | Aetna" in reader.prompts[-1]

    # correction of the employer: the derived user-level insurance fact
    # must vanish from the reader's context (the context fact about
    # acme's insurer legitimately stays - it was never wrong)
    mem.ingest_structured(
        [fact("user", "works_at", "apex", correction=True)], [],
        "2026-06-02")
    mem.answer("who insures the user?")
    assert "user | insured_by | Aetna" not in reader.prompts[-1]
    assert "user | works_at | apex" in reader.prompts[-1]


def test_empty_memory_answers_unknown_without_reader_call():
    mem, reader = build_memory()
    assert mem.answer("anything?") == "unknown"
    assert reader.prompts == []


def test_supersession_history_rendered_for_change_questions():
    mem, reader = build_memory()
    mem.ingest_structured([fact("user", "coffee_limit", "one cup")], [],
                          "2026-06-01")
    mem.ingest_structured([fact("user", "coffee_limit", "two cups")], [],
                          "2026-06-02")
    mem.answer("did the user's coffee limit increase?")
    prompt = reader.prompts[-1]
    assert "two cups" in prompt
    assert "previously: one cup" in prompt
