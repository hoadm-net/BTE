"""LLM-free tests for extraction schema construction, in particular the
relation_vocab enum constraint (schema-guided canonicalization for
domains with a known target relation vocabulary — see extraction.py's
module docstring for the Extract-Define-Canonicalize grounding)."""

from bte.extraction import RELATION_VOCAB_NOTE, _build_schema, _triple_props


def test_no_vocab_leaves_relation_as_free_string():
    props = _triple_props(None)
    assert props["relation"] == {"type": "string"}


def test_vocab_constrains_relation_to_enum():
    vocab = ["works_at", "lives_in"]
    props = _triple_props(vocab)
    assert props["relation"] == {"type": "string", "enum": vocab}
    # subject/object stay free-form - only relation is canonicalized
    assert props["subject"] == {"type": "string"}
    assert props["object"] == {"type": "string"}


def test_schema_applies_vocab_to_facts_premises_and_retractions():
    vocab = ["works_at", "lives_in"]
    schema = _build_schema(vocab)
    fact_props = schema["properties"]["facts"]["items"]["properties"]
    premise_props = fact_props["premises"]["items"]["properties"]
    retraction_props = schema["properties"]["retractions"]["items"]["properties"]
    for props in (fact_props, premise_props, retraction_props):
        assert props["relation"]["enum"] == vocab


def test_schema_without_vocab_matches_module_level_default():
    from bte.extraction import EXTRACTION_SCHEMA
    assert _build_schema(None) == EXTRACTION_SCHEMA


def test_vocab_note_formats_relation_list():
    note = RELATION_VOCAB_NOTE.format(relations="works_at, lives_in")
    assert "works_at, lives_in" in note


def test_fact_schema_constrains_domain_to_enum():
    from bte.extraction import DOMAINS
    schema = _build_schema(None)
    fact_props = schema["properties"]["facts"]["items"]["properties"]
    assert fact_props["domain"] == {"type": "string", "enum": DOMAINS}
    assert "domain" in schema["properties"]["facts"]["items"]["required"]
    # premises and retractions stay untyped: they are matched against
    # existing edges, which already carry their domain
    assert "domain" not in fact_props["premises"]["items"]["properties"]
    retraction_props = \
        schema["properties"]["retractions"]["items"]["properties"]
    assert "domain" not in retraction_props
