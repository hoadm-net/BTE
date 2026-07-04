# Evaluation Benchmarks

Each dataset below checks whether the failure mode predicted by Theorem 1
([formalism.md](formalism.md)) — single-edge systems cannot maintain JTS —
occurs in realistic conversational data, and whether the propagation algorithm
removes it.

## LongMemEval

[Wu et al., *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive
Memory*, ICLR 2025](https://arxiv.org/abs/2410.10813). 500 curated questions over
freely scalable user-assistant chat histories. Five ability categories; two
relevant here:

- **KU (Knowledge Update)**: the answer depends on the most recent update to a
  fact that was stated multiple times with different values across the
  conversation.
- **TR (Temporal Reasoning)**: the answer requires reasoning about *when* things
  happened relative to each other or to the question's reference time.

`LongMemEval-S` is the standard-scale variant used for general comparison; KU/TR
subsets are used for category-focused evaluation. The paper reports commercial
chat assistants and long-context LLMs drop ~30% accuracy under sustained
interaction, and proposes session decomposition + time-aware query expansion as
mitigations — neither involves multi-hop derived-fact dependency tracking.

**Scale (verified, for cost/runtime budgeting):** `-S` haystacks average ~115K
tokens across ~40–48 sessions (`-M` extends to ~1.5M tokens / ~500 sessions —
not used here). The exact KU/TR question count and whether their haystacks are
shared across questions (cheaper to extract once) or distinct per question
(linear in question count) is **not confirmed** from public summaries — check
the released JSON before budgeting extraction cost on this tier.

Note: **LongMemEval-V2** (arXiv [2605.12493](https://arxiv.org/html/2605.12493v1))
is a *different* benchmark despite the name — it targets agent *trajectory*
memory (web-agent environments, 25M–115M token histories), not conversational
KU/TR. It does fix a reader model (Qwen3.5-9B) to isolate memory-module quality —
that protocol idea is what this project borrows, but the KU/TR categories
themselves come from the original (v1) LongMemEval paper, not from V2.

## LoCoMo

[Maharana et al., *Evaluating Very Long-Term Conversational Memory of LLM
Agents*, ACL 2024](https://arxiv.org/abs/2402.17753). **10 conversations**
(19–32 sessions each, ~9K–16K tokens/conversation depending on
arXiv-vs-ACL-published version), generating **1,986 QA pairs** across five
categories: Single-hop, Multi-hop, Temporal, Open-domain, Adversarial.
Generated via an LLM-agent pipeline grounded on personas and temporal event
graphs. Used here for general non-regression / SOTA-comparison checks.

Corrected from an earlier draft of this doc/the project README, which stated
"~1540 conversations" — likely a conflation with the QA-pair count (1,986).
The corpus itself (10 conversations, ≲160K tokens total) is small; the cost
driver for this tier is the 1,986 QA pairs through reader+judge, not
extraction.

## BEAM

[*Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in
LLMs*, ICLR 2026](https://arxiv.org/abs/2510.27246)
([project page](https://mohammadtavakoli78.github.io/beam-light/)). 100
conversations, 2,000 validated probing questions, across four length tiers
(128K / 500K / 1M / 10M tokens) and 19 domains. Ten memory abilities; three
relevant here:

- **CR (Contradiction Resolution)**: conflicting statements planted across
  widely separated turns.
- **EO (Event Ordering)**: reconstructing the sequence of evolving information.
- **KU (Knowledge Update)**: detecting when a stated fact is later revised.

The paper does not distinguish *directly restated* contradictions from
*multi-hop derived* ones — CR questions are built from pairs of conflicting
plan bullets, not dependency chains. The authors report CR as the hardest
category across all evaluated methods, including their own LIGHT framework
(episodic memory + working memory + scratchpad), and state contradiction
detection "remains a challenging open problem" — i.e. unresolved even with
explicit memory architectures, supporting [H1](../README.md#hypotheses).

This project uses **BEAM-1M** only. BEAM-10M is excluded — cost per conversation
(10M tokens) is disproportionate to the marginal evaluation value for a single
module.

**Scale (verified):** 100 conversations / 2,000 questions total across 4
length tiers and 10 abilities. The exact per-tier (how many of the 100
conversations sit at the 1M tier) and per-ability (how many of the 2,000
questions are CR/EO/KU) breakdown is **not published in the abstract/summary
sources checked** — assume an even split (~25 conversations, ~150 CR/EO/KU
questions) only as a rough budgeting placeholder, and confirm against the
released dataset (`github.com/mohammadtavakoli78/BEAM`) before committing
budget.

## STALE

[*STALE: Can LLM Agents Know When Their Memories Are No Longer Valid?*,
2026](https://arxiv.org/abs/2605.06527). 400 expert-validated conflict
scenarios (1,200 queries) across 100+ everyday topics, contexts up to 150K
tokens. Three probing dimensions map closely onto this project's taxonomy
([contradiction-detection.md](contradiction-detection.md)):

- **State Resolution**: detecting a prior belief is outdated — close to
  *inherited conflict*.
- **Premise Resistance**: rejecting a query that presupposes a stale state —
  a downstream symptom of an un-propagated invalidation.
- **Implicit Policy Adaptation**: applying the updated state in behavior
  without being told to — requires the fact to actually be marked invalid,
  not just retrievable.

Best evaluated model/framework reaches only 55.2% overall — the strongest
available external evidence that the gap this project targets is real and
unsolved even by specialized memory frameworks, not only by raw LLMs. Adopted
as a category-focused evaluation tier (its own benchmark, not via category
subset extraction).

**Scale (verified):** 400 scenarios / 1,200 queries, contexts **"up to" 150K
tokens** — no published average. Cost estimates so far have assumed ~75K
tokens/scenario (half of the stated max) as a placeholder; this is the
single largest source of uncertainty in any cost projection involving STALE
and should be confirmed against the released data first.

## BeliefShift (related, not adopted)

[*BeliefShift: Benchmarking Temporal Belief Consistency and Opinion Drift in LLM
Agents*, 2026](https://arxiv.org/abs/2603.23848v1). 2,400 human-annotated
multi-session trajectories across health/politics/values/product-preference
domains; tracks Temporal Belief Consistency, Contradiction Detection, and
Evidence-Driven Revision. Closer to user *opinion* drift than asserted-fact
contradiction — noted here as related work, not adopted as an evaluation tier
(see [related-work.md](related-work.md)).

## Diagnostic probe (this project, not an external benchmark)

A synthetic set (~150–200 items) built specifically to isolate the effect of
multi-hop propagation, since none of the above benchmarks control hop depth
directly. Axes: hop depth (1–4), edge type (asserted vs. derived), temporal
density (same-session vs. distant), source confidence. Generated via templates
with LLM-assisted verification; a subset should be human-validated to avoid
generator/verifier circularity. Reported as analysis, not as a benchmark result.

## Harness note

Baselines (Zep, Mem0, Engram) are re-run on a shared retrieval/reader harness
rather than cited from vendor-published numbers — published Zep vs. Mem0
comparisons are inconsistent across sources, so fair comparison requires fixing
the reader and varying only the memory module (protocol borrowed from
LongMemEval-V2, see note above).
