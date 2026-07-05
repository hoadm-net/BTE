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

**Scale (confirmed against released data, checked 2026-07-05):** the
original HF release is deprecated; the maintained release is
`xiaowu0162/longmemeval-cleaned` (removes noisy history sessions that
interfered with answer correctness) — use it, since 2026 papers evaluate on
it. From `longmemeval_s_cleaned.json` directly: 500 questions — multi-session
133, **temporal-reasoning 133**, **knowledge-update 78**, single-session-user
70, single-session-assistant 56, single-session-preference 30; 30 of the 500
are abstention variants (`_abs` ids). Haystacks are effectively
**per-question**: only 20.5% of unique session ids appear in more than one
question, so extraction cost scales linearly — one full `-S` pass is ~61.2M
input tokens (median haystack ~123K tokens, ~48 sessions; chars/4 estimate).
KU questions carry exactly 2 evidence sessions each (old value, new value):
the update structure is 1-hop by construction, with no derived-fact chains —
consistent with the expectation that the diagnostic probe, not this tier,
carries the multi-hop diagnostic weight (H3's "more modest" real-world
prediction).

Note: **LongMemEval-V2** (arXiv [2605.12493](https://arxiv.org/html/2605.12493v1))
is a *different* benchmark despite the name — it targets agent *trajectory*
memory (web-agent environments, 25M–115M token histories), not conversational
KU/TR. It does fix a reader model (Qwen3.5-9B) to isolate memory-module quality —
that protocol idea is what this project borrows, but the KU/TR categories
themselves come from the original (v1) LongMemEval paper, not from V2.

## LoCoMo

[Maharana et al., *Evaluating Very Long-Term Conversational Memory of LLM
Agents*, ACL 2024](https://arxiv.org/abs/2402.17753). **10 conversations**
(19–32 sessions each), **1,986 QA pairs** across five categories:
Single-hop, Multi-hop, Temporal, Open-domain, Adversarial. Generated via an
LLM-agent pipeline grounded on personas and temporal event graphs. Used here
for general non-regression / SOTA-comparison checks.

**Scale (confirmed against released `locomo10.json`, checked 2026-07-05):**
10 conversations / 1,986 QA exactly. Per-category-id counts: 1: 282, 2: 321,
3: 96, 4: 841, 5: 446 (the id→name mapping is not stated in the JSON;
by size, 4 = single-hop and 5 = adversarial are the natural reading —
verify against the paper appendix before naming categories in the paper).
Measured ~10.8–22.2K tokens/conversation, ~180K total (chars/4 estimate) —
slightly above the ~9K–16K cited from the paper, which counts tokens with a
real tokenizer on the ACL version. Either way the corpus is small; the cost
driver for this tier remains the 1,986 QA pairs through reader+judge, not
extraction.

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

**Scale (confirmed against released data, checked 2026-07-05):** per-tier
split from the repo README: 128K: 20, 500K: 35, **1M: 35**, 10M: 10
conversations. From the released `probing_questions.json` files, the 1M tier
has exactly 2 questions per ability per conversation: **700 questions, 70
per ability — CR + EO + KU = 210** (the earlier ~150 placeholder was low).
Measured conversation size at the 1M tier: 0.78–1.93M tokens, median 1.14M,
~42M tokens total (chars/4 on the released JSON) — one full ingestion pass
of the 1M tier is comparable in input volume to a LongMemEval-S pass.

Two harness-relevant observations from the released CR items: (i) they are
direct restated contradictions, as the paper says — `contradiction_type`
labels like `never_statement_violation`, with `source_chat_ids` pointing at
both sides; (ii) the `ideal_answer` is to *flag the contradiction and ask
for clarification*, not to pick a winner. A system that silently resolves
by recency (ours, and most baselines) will fail BEAM-CR's rubric unless the
answer layer surfaces detected conflicts — the retrieval/reader adapter
must expose conflict metadata, not just the winning edge. Plan for this in
the harness before E3, or BEAM-CR will under-credit exactly the systems
that handle contradictions best.

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

**Scale (confirmed against released data, checked 2026-07-05):** release at
`github.com/icedreamc/STALE` + HF `STALEproj/STALE`
(`T1_T2_400_FULL.json`, 306MB). 400 scenarios / 1,200 queries exactly
(3 per scenario, one per probing dimension). The scenarios split 200/200
into **T1 (explicit update** — the new state is stated outright**) and T2
(implicit update** — the new state must be inferred, e.g. a Portland
resident who starts mentioning bark scorpions and dry heat**)**; T2 is the
extraction-difficulty half, directly relevant to the `classify(e)`
reliability question. Each scenario carries a 50-session haystack with
`relevant_session_index` marking the old/new-state sessions — usable as
gold for internal-state scoring, not just end-to-end QA. Measured context
size: 162–194K tokens/scenario, median ~179K, ~71.4M total (chars/4 on the
full JSON; a real tokenizer will land somewhat lower) — the earlier ~75K
placeholder understates this tier's cost by ~2x; redo the cost projection
before E3. The HF repo ships a `LongMemEval_LICENSE`: STALE haystacks build
on LongMemEval session material, so treat LongMemEval and STALE tiers as
sharing surface distribution (not independent evidence) when interpreting
results across the two.

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
