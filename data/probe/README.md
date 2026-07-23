# Diagnostic Probe — Annotator Guide

This directory contains the diagnostic probe used to evaluate the
"Beyond the Edge" (BTE) research project's core claim: that propagating
a contradiction across a multi-hop derivation chain (not just at the
single fact where it was stated) improves a conversational memory
system's accuracy. This document explains what the probe is, how it
was built, and what we need you to check before we treat it as final.

## 1. What this is for

Conversational AI systems keep a memory of things a user has told them
across many sessions. Sometimes a user's later statement contradicts
something established earlier — directly ("I don't live in Madison
anymore, I moved to Tucson") or indirectly, by contradicting something
that was only *inferred* from an earlier statement, several steps down
a chain of reasoning.

Most public benchmarks for this don't let a researcher control how
many reasoning steps ("hops") separate the contradiction from the fact
it should invalidate. This probe does: it is a set of short, synthetic
multi-session conversations, generated so that hop depth (1 to 4),
along with four other properties, is a controlled variable rather than
an accident of whatever text happened to be scraped.

Each item is graded by asking a memory system a single question before
and after the contradicting turn, and comparing its answer to a gold
answer we compute at generation time. Your job is not to run any
system — it is to confirm the *conversation and the gold answers make
sense on their own*, before any system ever sees them.

## 2. How it was built

The generator lives in `bte/probe.py` (`generate()`). It is fully
deterministic — same seed (`20260705`) in, same 168 items out, no
randomness at read time and no LLM involved in writing the
conversational turns. Every sentence in every item comes from a fixed
Python string template filled in with domain-specific values, not from
paraphrasing. That is deliberate: it keeps the *content* of what's
being tested fixed and auditable, at the cost of the turns reading a
little repetitive across items in the same domain — that repetition is
expected, not a bug to flag.

**Design axes** (every combination is generated, minus the one
impossible cell — a depth-1 item has no derived fact yet to
contradict):

| Axis | Values | Meaning |
| --- | --- | --- |
| Hop depth | 1–4 | How many reasoning steps separate the root fact from the fact actually being asked about |
| Contradicted edge | asserted / derived | Does the contradiction target the root fact itself, or a conclusion several hops downstream? |
| Axis | update / correction | Did the world change ("I moved"), or was the original statement simply wrong ("I never lived there")? |
| Density | adjacent / distant | Are the chain-building turns back to back, or spread across sessions with unrelated filler turns in between? |
| Confidence | high / low | Is the root fact stated plainly, or hedged ("I think...", "if I remember right...")? |

**Domains**: the same abstract 4-hop structure is realized in three
unrelated everyday-life surface stories — `employment` (job → office
city → timezone → daylight-saving policy), `residence` (city →
electric utility → billing portal → portal owner), and `training`
(race → training plan → long-run day → blocked calendar slot). Each of
the 3 generation replicates is pinned to one domain, so every
(domain, hop depth) combination appears with only one replicate's worth
of items — this is intentional, not a sampling gap.

**Gold labels** (`gold_pre`, `gold_post`, `gold_invalidated`,
`gold_axis` in each item) are computed by the same generator code from
the same template values used to write the conversation — they are
"correct by construction" in the sense that the code that writes the
question also writes the answer, from the same source values. What
that construction *cannot* catch on its own is whether the natural-
language sentence a human reads actually conveys what the generator
intended, unambiguously. That gap is real: this generator went through
several rounds of revision after we ran real systems against it and
found specific items where the wording supported a second, unintended
reading (for example, a sentence meant to say "the city's utility
company is X" was worded in a way a careful reader could also parse as
"X is located in the city," which is a different, unintended claim).
Structural bugs like that are now fixed, but they were only found by
close reading — which is exactly what we're asking you to do now, as an
independent check before we treat this set as final.

## 3. What each item looks like

`probe_v0.json` is a JSON array of 168 items. Each item has:

- `probe_id` — short identifier encoding the axis values, e.g.
  `emp-d3-der-corr-adj-hi-r0` (employment, depth 3, derived edge
  contradicted, correction, adjacent density, high confidence).
- `sessions` — a list of conversation turns (each turn a list of one
  user message). Read these in order; this is the entire conversation
  the memory system would see.
- `question` — the single question asked both before and after the
  final (contradicting) turn.
- `gold_pre` — the correct answer to `question` using only the turns
  *before* the last one.
- `gold_post` — the correct answer *after* the last turn is included.
  This is sometimes the literal string `"unknown"` — that's correct
  whenever the contradiction removes the basis for the old answer
  without stating what the new one is.
- `gold_invalidated` — which relations should no longer be trusted
  after the contradiction (informational; you don't need to verify the
  internal relation names, just that the idea "these facts are now
  stale" matches your own reading of the conversation).
- `domain`, `hop_depth`, `contradicted`, `axis`, `density`, `confidence`
  — the design-axis values described above, included for reference.

You can ignore `oracle_facts` / `oracle_retractions` — those are an
internal structured form used for a separate automated consistency
check, not something to hand-verify.

## 4. What to check, per item

For each item, read `sessions` in order as if you were
the memory system, then answer these, in order — if an earlier check
fails, later ones may not apply (note that and move on):

1. **Readable**: do the turns read as plausible things a user might
   type, even if plainly-templated? Flag anything broken, garbled, or
   self-contradictory *within* a single turn.
2. **Chain sound** (hop depth ≥ 2 only): does each turn's fact connect
   to the previous one — i.e. does turn 2 describe a property of the
   *exact* entity turn 1 introduced, not a different or ambiguous one?
3. **Contradiction unambiguous**: does the final turn clearly target
   *one specific* earlier fact, under the one reading a careful human
   would give it? This is the main thing we need checked — if you can
   construct a second, equally-plausible reading of the final turn
   that points somewhere else (or nowhere), that's a fail, and please
   write down the alternate reading you found.
4. **`gold_pre` correct**: given only the turns before the last one,
   is `gold_pre` the right answer to `question`?
5. **`gold_post` correct**: given all the turns, is `gold_post` the
   right answer? If it's `"unknown"`, confirm that's actually right —
   i.e. that nothing in the conversation supplies a replacement value.
6. **Question fit**: is `question` unambiguous about which fact it's
   asking for (no plausible reading where it's asking about something
   else the conversation also mentions)?

**Verdict per item**: PASS or FAIL. For FAIL, write one line naming
which check (1–6) failed and why — that's what lets us regenerate the
right thing instead of guessing.

## 5. Coverage and reporting

The set is small enough to review in full - go through all 168 items
rather than sampling. The `annotator/` tool (see its README) loads the
whole set and tracks your progress across sessions, so there's no need
to plan a subset in advance.

Report back:

- number of items reviewed, number PASS, number FAIL (this is the
  headline agreement number we need);
- the FAIL list: `probe_id`, which check number failed, and a one-line
  reason;
- anything you noticed that felt off but didn't cleanly fail one of
  the six checks above — worth a note even without a formal verdict.

## 6. Provenance

`probe_v0.json` and `probe_v0.json.sha256` are regenerated by
`uv run python pilot/generate_probe.py` and are deterministic from the
seed in `bte/probe.py`. If the checksum in `probe_v0.json.sha256`
doesn't match `sha256sum probe_v0.json`, the file was hand-edited or
corrupted after generation — flag that rather than reviewing it.
