"""LLM-free end-to-end memory tests: structured facts in, reader prompt
out — checks that what BBP supersedes is never presented as a live,
current fact (the property Mem0's smoke test failed on). A derived fact
invalidated by propagation (no successor edge states a replacement) is
shown to the reader as an explicit INVALIDATED marker rather than
silently dropped - a prior version dropped it entirely, which let the
reader reconstruct the exact retired conclusion from the still-active
raw facts that had fed it.
"""

from bte.graph import BJG, Edge
from bte.ingest import Ingestor
from bte.lattice import S, Sigma
from bte.memory import BJGMemory, render_fact
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
    assert "- user | insured_by | Aetna" in reader.prompts[-1]

    # correction of the employer: the derived user-level insurance fact
    # has no successor to carry a "(previously: X)" note (nothing
    # states a new insurer), so it must appear as an explicit
    # INVALIDATED marker - never again as a plain, live fact line (the
    # context fact about acme's insurer legitimately stays untouched -
    # it was never wrong)
    mem.ingest_structured(
        [fact("user", "works_at", "apex", correction=True)], [],
        "2026-06-02")
    mem.answer("who insures the user?")
    assert "- user | insured_by | Aetna" not in reader.prompts[-1]
    assert "INVALIDATED: user | insured_by | Aetna" in reader.prompts[-1]
    assert "user | works_at | apex" in reader.prompts[-1]


def test_render_fact_marks_inactive_edge_as_invalidated():
    g = BJG()
    g.add_asserted(Edge(id="e1", subject="user", relation="works_at",
                        object="Acme", t_transaction="2026-01-01"))
    g.force_status("e1", Sigma(S.TOP, S.BOT))
    line = render_fact(g.edges["e1"], graph=g)
    assert line.startswith("- INVALIDATED: user | works_at | Acme")
    assert "no longer holds" in line
    assert "recorded 2026-01-01" in line


def test_render_fact_normal_for_active_edge():
    g = BJG()
    g.add_asserted(Edge(id="e1", subject="user", relation="works_at",
                        object="Acme", t_transaction="2026-01-01"))
    line = render_fact(g.edges["e1"], graph=g)
    assert line == "- user | works_at | Acme (recorded 2026-01-01)"


def test_bare_retraction_with_no_replacement_shown_as_invalidated():
    """A retraction with no new value (e.g. "forget that I said X", was
    never true) leaves NO successor edge either - the same gap as
    BBP-propagated derived-fact invalidation, just via a different
    path (ingest.py's apply_retractions calls bbp() directly with no
    new edge created)."""
    mem, reader = build_memory()
    mem.ingest_structured([fact("user", "coffee_limit", "one cup")], [],
                          "2026-06-01")
    mem.ingest_structured([], [
        {"subject": "user", "relation": "coffee_limit", "object": "one cup",
         "was_wrong": True},
    ], "2026-06-02")
    mem.answer("what is the user's coffee limit?")
    assert "INVALIDATED: user | coffee_limit | one cup" in reader.prompts[-1]
    assert "- user | coffee_limit | one cup (recorded" not in reader.prompts[-1]


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
    assert "recorded 2026-06-02" in prompt  # current fact's own record date


def test_recorded_date_and_current_date_exposed_for_date_arithmetic():
    """A date-elapsed question needs both when the fact was recorded and
    what 'today' is - neither was visible to the reader before (bench.py
    30-question slice: 6/14 temporal-reasoning misses were pure date
    arithmetic with the needed fact already retrieved)."""
    mem, reader = build_memory()
    mem.ingest_structured(
        [fact("user", "started_lessons_for", "ukulele")], [], "2023-02-01")
    mem.answer("How long ago did the user start ukulele lessons?",
              reference_time="2023-04-01")
    prompt = reader.prompts[-1]
    assert "Current date: 2023-04-01" in prompt
    assert "recorded 2023-02-01" in prompt


def test_no_current_date_line_when_reference_time_absent():
    mem, reader = build_memory()
    mem.ingest_structured([fact("user", "lives_in", "Seattle")], [],
                          "2026-06-01")
    mem.answer("where does the user live?")
    assert "Current date:" not in reader.prompts[-1]
