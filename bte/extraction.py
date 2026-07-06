"""Turn-to-edges extraction (proposed-model.md section 2, first stage).

Two outputs per message: new positive facts (with premises when the
sentence derives them from other facts — the hook that lets stated
consequences enter the graph as derived edges), and retractions — known
facts the message declares ended or wrong. Keeping retractions out of the
facts list is deliberate: an earlier single-list design made the model
mark replacement facts as retractions and lose them.
"""

from __future__ import annotations

import re

from .llm import CachedLLM

_TRIPLE_PROPS = {
    "subject": {"type": "string"},
    "relation": {"type": "string"},
    "object": {"type": "string"},
}

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    **_TRIPLE_PROPS,
                    "valid_from": {"type": ["string", "null"]},
                    "valid_to": {"type": ["string", "null"]},
                    "confidence": {"type": "number"},
                    "is_correction": {"type": "boolean"},
                    "premises": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": _TRIPLE_PROPS,
                            "required": ["subject", "relation", "object"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["subject", "relation", "object", "valid_from",
                             "valid_to", "confidence", "is_correction",
                             "premises"],
                "additionalProperties": False,
            },
        },
        "retractions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    **_TRIPLE_PROPS,
                    "was_wrong": {"type": "boolean"},
                },
                "required": ["subject", "relation", "object", "was_wrong"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["facts", "retractions"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """Extract durable user facts from the message, and identify which known facts the message retracts.

Output two lists:

1. facts — NEW positive facts stated in the message.
   - subject: "user" for the speaker; otherwise the named entity.
   - relation: short snake_case predicate. Reuse the relation names and
     entity spellings of the known facts whenever the new fact concerns
     the same kind of thing.
   - valid_from/valid_to: ISO dates if stated, else null.
   - confidence: 1.0 for direct statements, lower for hedged ones.
   - is_correction: true only if the speaker says an earlier statement
     was WRONG (misspoke, never true) and this fact replaces it.
   - premises: triples this fact is explicitly derived from ("since X",
     "because Y") — copy them exactly from the known facts when they match.
   - Standing constraints and preferences ("never schedule X on...",
     "always Y") are durable facts. Questions and one-off requests are not.
   - Never put a denied or ended fact here.

2. retractions — known facts that the message says no longer hold or
   never held, copied EXACTLY from the known facts list.
   - was_wrong: true if the fact was wrong from the start (correction),
     false if the world changed (the fact ended).
   - Only list facts that appear in the known facts list. Empty if none."""


def normalize(term: str) -> str:
    term = term.strip().lower()
    term = re.sub(r"[^a-z0-9]+", "_", term)
    return term.strip("_")


def _norm_triple(t) -> dict | None:
    """Models occasionally emit null entries or null fields inside the
    facts array; drop anything that is not a complete triple."""
    if not isinstance(t, dict):
        return None
    s, r, o = t.get("subject"), t.get("relation"), t.get("object")
    if not (isinstance(s, str) and isinstance(r, str)
            and isinstance(o, str) and s and r and o.strip()):
        return None
    t["subject"] = normalize(s)
    t["relation"] = normalize(r)
    t["object"] = o.strip()
    return t


def extract_facts(llm: CachedLLM, text: str, reference_date: str,
                  context: list[str] | None = None) -> dict:
    known = ""
    if context:
        known = "Known active facts:\n" + "\n".join(
            f"- {c}" for c in context) + "\n"
    data = llm.complete_json(
        SYSTEM_PROMPT,
        f"Reference date: {reference_date}\n{known}Message: {text}",
        "extraction",
        EXTRACTION_SCHEMA,
    )
    facts = []
    for f in data.get("facts") or []:
        if _norm_triple(f) is None:
            continue
        f["premises"] = [p for p in (f.get("premises") or [])
                         if _norm_triple(p) is not None]
        facts.append(f)
    retractions = [r for r in data.get("retractions") or []
                   if _norm_triple(r) is not None]
    return {"facts": facts, "retractions": retractions}
