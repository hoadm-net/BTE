# Models Used in the Pipeline

[proposed-model.md](proposed-model.md) defines the pipeline roles (extraction,
adjudication, reader, judge, embedding) but deliberately leaves the concrete
model choice unspecified — that's an infrastructure decision, not part of the
contribution. This doc records the actual choice per role, why, and the
pricing each was checked against (06/2026).

## Role → model

| Role | Model | Price / 1M tok (in / out, 06/2026) | Why |
| --- | --- | --- | --- |
| Extraction (turn → edges) | **candidate set: DeepSeek-V3.2 (leaning default), gpt-4o-mini, GPT-4.1 nano** — final pick by probe bake-off, see 07/2026 revision below | $0.229/$0.343, $0.15/$0.60, $0.10/$0.40 | quality per dollar + reproducibility now outrank raw price; decided by measured edge-F1, not assertion |
| Conflict adjudication (escalate step, [contradiction-detection.md](contradiction-detection.md)) | same candidate set, same bake-off | same | same call shape (read context, emit small judgment) |
| Reader (fixed, [proposed-model.md](proposed-model.md) §5) | **Qwen3.5-9B** | $0.10 / $0.15 | fixed-reader protocol borrowed from LongMemEval-V2, open-weight, cheaper than any closed option for this role |
| Judge | **GPT-5** (full, not mini/nano) | $0.63 / $5.00 | replaces GPT-4o — see below |
| Embedding (BJG retrieval channel) | **Qwen3-Embedding-8B** | $0.01 / $0 | open-weight, pairs with Qwen3.5-9B per LongMemEval-V2's own memory-controller setup |

No hardware constraint applies to any of these — all are accessed via API
(OpenRouter for the open-weight models), not self-hosted, per the earlier
decision to avoid GPT-OSS-120b's hardware requirement.

## Why GPT-5 (not GPT-4o) for judge

GPT-4o is now **grandfathered legacy pricing** ($2.50/$10.00) — OpenAI's
current flagship is the 5.5/5.4 line, and the original **GPT-5** sits at
$0.63/$5.00, roughly 4x cheaper than GPT-4o for a newer-generation model.
Judge is a low-volume role (one call per question, not per turn), so the
cheaper+stronger swap is a strict win with no real downside. **DeepSeek-V3.2**
($0.229/$0.343, Engram's own choice) remains the cheapest judge option if
budget is the binding constraint over judge quality — worth a sensitivity run,
not the default.

## Why *not* GPT-5-mini/nano for extraction & adjudication

The premise "newer GPT-5 generation, mini/nano tiers, therefore cheaper" turns
out to be only partly true — OpenAI raised **output** pricing faster than it
lowered input pricing across the 5.4 dot-release:

| Model | Input $/1M | Output $/1M |
| --- | --- | --- |
| gpt-4o-mini (current pick) | $0.15 | $0.60 |
| GPT-5 nano (original, Aug 2025) | $0.05 | $0.40 |
| GPT-5.4 nano (current) | $0.20 | $1.25 |
| GPT-5 mini | $0.13 | $1.00 |
| GPT-5.4 mini | $0.75 | $4.50 |

Extraction/adjudication is input-heavy, low-output (read a long context, emit
a small structured edge). On the STALE-tier cost profile modeled earlier
(~30M input / ~4.5M output tokens for one full pass), the total cost ranks:

- gpt-4o-mini: **$7.20**
- GPT-5 mini: $8.40
- GPT-5.4 nano: $11.63
- GPT-5.4 mini: $42.75
- GPT-5 nano (original): **$3.30** ← only option cheaper than the current pick

So every *current* (5.4-generation) mini/nano tier is actually more expensive
than gpt-4o-mini for this call shape, because their output price went up more
than their input price went down. The one genuinely cheaper option is the
**original GPT-5 nano** (Aug 2025 dot-release) — but that's an older release
within the same family, the first candidate OpenAI would deprecate, so its
continued availability shouldn't be assumed without checking
`developers.openai.com/api/docs/pricing` directly at run time.

**Recommendation:** keep gpt-4o-mini as the default for extraction/
adjudication. Treat GPT-5 nano (original) as a cost experiment — re-run the
diagnostic probe tier (cheap, ~175 items) on it first to confirm both the
price advantage and that the older nano's extraction quality doesn't regress
JTS-relevant accuracy, before switching the full pipeline.

## 07/2026 revision: extraction/adjudication becomes a measured bake-off

Pricing re-checked 2026-07-05 (developers.openai.com/api/docs/pricing,
openrouter.ai), after the pilot confirmed real volumes: one full extraction
pass over all four tiers is ~175M input / ~26M output tokens (LongMemEval-S
61M + BEAM-1M 42M + STALE 71M + LoCoMo; see
[evaluation-benchmarks.md](evaluation-benchmarks.md)).

| Model | In / Out $/1M (07/2026) | Full pass | Notes |
| --- | --- | --- | --- |
| GPT-5.5 (flagship) | $5.00 / $30.00 | ~$1,655 | no 5.5 mini/nano tiers exist as of 07/2026; Batch/Flex halves it; cached input $0.50 |
| GPT-5.4 nano | $0.20 / $1.25 | ~$68 | output price still uncompetitive for this call shape |
| gpt-4o-mini | $0.15 / $0.60 | ~$42 | legacy but still served; Zep's own extraction choice |
| GPT-4.1 nano | $0.10 / $0.40 | ~$28 | cheapest closed option currently served |
| DeepSeek-V3.2 (OpenRouter) | $0.229 / $0.343 | ~$49 | ~236B-class open weights |

At these volumes the bulk-role cost spread is ~$20–40 per pass — not a
binding constraint. The decision criteria are therefore (in order):
extraction quality (a missed edge poisons every downstream justification
chain — direct H2 risk), reproducibility (open weights can be re-run by
reviewers indefinitely; closed minis carry deprecation risk, cf. the GPT-5
nano case below), and only then price. **Decision: run all three candidates
over the diagnostic probe when it exists (edge-level F1 against the probe's
gold justification chains, ~$5 total) and pick on measured quality; default
leaning DeepSeek-V3.2 on the quality-per-dollar and open-weights arguments.
The bake-off table goes in the paper's appendix** — it doubles as the
extractor-sensitivity ablation reviewers ask for. GPT-5.5 is excluded for
bulk roles on cost-benefit (40x price for a read-long/emit-short task);
recorded here so the exclusion is a documented comparison, not an omission.

Two cost mechanics worth using regardless of winner: OpenAI Batch API
(-50%, extraction passes are offline-friendly; OpenRouter has no batch
tier as of 07/2026), and the disk-level LLM call cache in the harness plan.

The 06/2026 GPT-5-nano cost experiment below is superseded by this bake-off
(kept for the pricing history).

## Open question

No model is fixed yet for `classify(e)` (axis disambiguation — valid-time vs.
transaction-time, [proposed-model.md](proposed-model.md) §3). Likely folds
into the same adjudication call (gpt-4o-mini) rather than a separate model,
but not yet decided.
