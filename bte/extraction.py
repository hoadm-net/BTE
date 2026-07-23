"""Turn-to-edges extraction (proposed-model.md section 2, first stage).

Two outputs per message: new positive facts (with premises when the
sentence derives them from other facts — the hook that lets stated
consequences enter the graph as derived edges), and retractions — known
facts the message declares ended or wrong. Keeping retractions out of the
facts list is deliberate: an earlier single-list design made the model
mark replacement facts as retractions and lose them.

Relation-name canonicalization (optional `relation_vocab`): this is the
same problem as surface-form variation in Open Information Extraction —
the same relation coming out as "power_utility_of" in one call and
"electric_utility_is" in another, which silently breaks ChainRule
matching in rules.py (derive_closure requires exact relation equality, so
a justification edge that should exist just never gets built — a
structural gap, not a BBP failure; BBP is proven correct over whatever
graph it is handed, see formalism.md Theorem 2). When the caller knows the
target relation vocabulary in advance (the diagnostic probe's domains have
one), this is the schema-guided case of Extract-Define-Canonicalize's
"Target Alignment" (Zhang et al., EMNLP 2024,
https://arxiv.org/abs/2404.03868) — EDC does target alignment as a
post-hoc cosine-similarity + LLM-verification pass; here it is done
structurally instead, via a JSON-schema `enum` constraint on the relation
field, which is a strictly stronger guarantee (the model cannot emit an
off-vocabulary relation at all, not just "is discouraged from it").
Open-domain data with no fixed target schema (LongMemEval, STALE) doesn't
have this option — canonicalizing that is a separate, harder problem
(Zep/Graphiti's entropy-gated fuzzy entity/edge matching is the relevant
precedent; not implemented here, tracked as follow-up work).
"""

from __future__ import annotations

import re
from typing import Optional

from .llm import CachedLLM

# Coarse life-domain types for extracted facts. This is the top level of a
# two-level (domain, relation) typing: the relation plays the fine-grained
# slot role, the domain gates cross-relation conflict candidate generation
# (see conflict.DomainDependencies). Written as a generic everyday-life
# ontology, independent of any benchmark's generation ontology; CUPMem
# (STALE, arXiv 2605.06527, Appendix F) uses the same two-level shape but
# with a fixed hand-authored slot layer and no online learning.
DOMAINS = [
    "health",
    "work_or_study",
    "schedule_and_routine",
    "location_and_residence",
    "transport_and_commute",
    "family_and_social",
    "finance_and_resources",
    "possessions_and_devices",
    "preferences_and_habits",
    "plans_and_goals",
    "other",
]


def _triple_props(relation_vocab: Optional[list[str]] = None) -> dict:
    relation_schema: dict = {"type": "string"}
    if relation_vocab:
        relation_schema = {"type": "string", "enum": list(relation_vocab)}
    return {
        "subject": {"type": "string"},
        "relation": relation_schema,
        "object": {"type": "string"},
    }


def _build_schema(relation_vocab: Optional[list[str]] = None) -> dict:
    triple_props = _triple_props(relation_vocab)
    return {
        "type": "object",
        "properties": {
            "facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        **triple_props,
                        "domain": {"type": "string", "enum": DOMAINS},
                        "valid_from": {"type": ["string", "null"]},
                        "valid_to": {"type": ["string", "null"]},
                        "confidence": {"type": "number", "minimum": 0,
                                      "maximum": 1},
                        "is_correction": {"type": "boolean"},
                        "premises": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": triple_props,
                                "required": ["subject", "relation", "object"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["subject", "relation", "object", "domain",
                                 "valid_from", "valid_to", "confidence",
                                 "is_correction", "premises"],
                    "additionalProperties": False,
                },
            },
            "retractions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        **triple_props,
                        "was_wrong": {"type": "boolean"},
                    },
                    "required": ["subject", "relation", "object",
                                 "was_wrong"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["facts", "retractions"],
        "additionalProperties": False,
    }


EXTRACTION_SCHEMA = _build_schema()

SYSTEM_PROMPT = """Extract durable user facts from the message, and identify which known facts the message retracts.

Output two lists:

1. facts — NEW positive facts stated in the message.
   - subject: "user" for the speaker; otherwise the named entity.
   - relation: a short snake_case ATTRIBUTE name, specific enough that
     (subject, relation) works as a slot whose current value is the
     object. Never use a bare modal or generic verb as the whole
     relation - "plans_to", "needs_to", "has", "is", "wants" are wrong;
     "plans_to_reduce_screen_time", "daily_commitment_cap",
     "wake_up_time" are right. When the new fact updates an attribute
     that already appears in the known facts, copy that relation name
     EXACTLY instead of inventing a variant.
   - object: a concrete, self-contained value describing the actual state.
     NEVER a bare "true"/"false" and NEVER a repeat of the relation name
     (e.g. relation "observes_dst" with object "observes_dst" is wrong).
     For yes/no or on/off facts, phrase the object as the state itself in
     a few words: relation "observes_dst", object "observes daylight
     saving time" when true, object "stays on standard time, no daylight
     saving" when false. A reader must be able to understand the object
     without negating the relation name.
   - domain: the life area the fact belongs to, from the fixed list in
     the schema. Pick the area the fact is ABOUT (a job's hours are
     work_or_study; a habit of limiting commitments for health reasons
     is health). Use "other" only when nothing fits.
   - valid_from/valid_to: ISO dates if stated, else null.
   - confidence: a number from 0 to 1, always. 1.0 for direct statements
     (including plain statements about third parties, e.g. "Acme Corp's
     office is in Denver" is confidence 1.0 - it is not hedged just
     because the subject isn't the speaker). Lower only for actually
     hedged statements ("I think", "if I remember right"). Never a
     placeholder or negative value.
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

RELATION_VOCAB_NOTE = """
Every "relation" field (in facts, premises, and retractions) MUST be
chosen from this fixed list — pick whichever one the sentence actually
expresses, do not invent a new name even if it seems more natural:
{relations}"""


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
                  context: list[str] | None = None,
                  relation_vocab: list[str] | None = None) -> dict:
    known = ""
    if context:
        known = "Known active facts:\n" + "\n".join(
            f"- {c}" for c in context) + "\n"
    system = SYSTEM_PROMPT
    schema = EXTRACTION_SCHEMA
    if relation_vocab:
        system = SYSTEM_PROMPT + RELATION_VOCAB_NOTE.format(
            relations=", ".join(relation_vocab))
        schema = _build_schema(relation_vocab)
    data = llm.complete_json(
        system,
        f"Reference date: {reference_date}\n{known}Message: {text}",
        "extraction",
        schema,
    )
    facts = []
    for f in data.get("facts") or []:
        if _norm_triple(f) is None:
            continue
        if f.get("domain") not in DOMAINS:
            f["domain"] = "other"
        f["premises"] = [p for p in (f.get("premises") or [])
                         if _norm_triple(p) is not None]
        facts.append(f)
    retractions = [r for r in data.get("retractions") or []
                   if _norm_triple(r) is not None]
    return {"facts": facts, "retractions": retractions}
