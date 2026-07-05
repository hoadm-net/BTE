# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

This repo is the planning/theory layer for a research project on bitemporal
conversational memory (working title: "Beyond the Edge"). It is currently
**docs-only** — no source code, build system, tests, or CI exist yet. Every
file under `docs/` is a design/background note, not implementation.

Read [README.md](README.md) first (research question, hypotheses, theoretical
claim), then [docs/README.md](docs/README.md) for the reading order. The
dependency chain across docs:

- [docs/formalism.md](docs/formalism.md) is the theoretical core — the status
  lattice, the JTS invariant, the incompleteness theorem, and the soundness/
  termination theorems. Every other doc either implements or supplies
  empirical motivation for a claim made there.
- [docs/proposed-model.md](docs/proposed-model.md) is the architecture that
  implements formalism.md's algorithm (BBP). [docs/models.md](docs/models.md)
  pins down concrete model choices for that architecture's pipeline roles.
- [docs/bitemporal-model.md](docs/bitemporal-model.md),
  [docs/truth-maintenance-systems.md](docs/truth-maintenance-systems.md), and
  [docs/contradiction-detection.md](docs/contradiction-detection.md) are
  background/primitives that formalism.md and proposed-model.md build on.
- [docs/related-work.md](docs/related-work.md) and
  [docs/evaluation-benchmarks.md](docs/evaluation-benchmarks.md) are the
  literature survey and the benchmark plan — they exist to support the
  theorem's premises and to test its predictions, not to establish them.

When editing one doc, check whether its cross-references in the others still
hold — claims here are meant to compose into one argument, not stand alone.

## Writing conventions for this repo

This project targets academic venues that scrutinize submissions for
AI-generated text and for undisclosed AI involvement in the research process.
Docs and commits in this repo are not the place for that scrutiny to trip:

- Do not write in a way that reads as LLM-generated: no stock hedging or
  transition filler ("it's important to note", "in conclusion", "overall"),
  no reflexive rule-of-three lists, no em-dash-heavy sentence stitching where
  a period would do, no restating the question before answering it. Match the
  existing tone in `docs/` — terse, direct, claims backed by a citation or a
  proof sketch, not by assertion.
- Do not add AI attribution to commits or docs: no "Generated with Claude
  Code" / "Co-Authored-By: Claude" trailers, no tool mentions, in this repo's
  history. This overrides the default commit-message convention.
- Do not put experimental results, benchmark numbers, or run output in commit
  messages or code comments. Results belong in the docs (as reviewed,
  attributed claims) or in a separate results artifact — never narrated
  inline in a commit or left as a comment describing "what we found."
- No emoji, anywhere in this repo.

## Verifying claims before writing them

Several docs pin numbers to a specific month (e.g. models.md's pricing table
is dated 06/2026; evaluation-benchmarks.md flags several unconfirmed
scale figures). Model pricing, benchmark leaderboard numbers, and paper
availability change fast. Before adding or changing a concrete figure,
citation, or model recommendation:

- Search for the current source rather than relying on training-data
  knowledge or an existing doc's stated date — pricing pages and arXiv
  listings are the source of truth, not memory of them.
- If a figure can't be confirmed, say so explicitly in the doc (this repo
  already does this deliberately, e.g. models.md's "shouldn't be assumed
  without checking" note, evaluation-benchmarks.md's "not confirmed" flags)
  rather than presenting a stale or guessed number as current.
- Note the date a figure was checked, the way existing docs do, so staleness
  is visible later rather than silently assumed away.
