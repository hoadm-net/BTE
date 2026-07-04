# Models Used in the Pipeline

[proposed-model.md](proposed-model.md) defines the pipeline roles (extraction,
adjudication, reader, judge, embedding) but deliberately leaves the concrete
model choice unspecified — that's an infrastructure decision, not part of the
contribution. This doc records the actual choice per role, why, and the
pricing each was checked against (06/2026).

## Role → model

| Role | Model | Price / 1M tok (in / out, 06/2026) | Why |
| --- | --- | --- | --- |
| Extraction (turn → edges) | **gpt-4o-mini** | $0.15 / $0.60 | matches Zep's choice for graph construction ([related-work.md](related-work.md) §1); cheapest option for an input-heavy/output-light task once GPT-5.x mini/nano tiers are checked (see below) |
| Conflict adjudication (escalate step, [contradiction-detection.md](contradiction-detection.md)) | **gpt-4o-mini** | same | same reasoning; same call shape (read context, emit small judgment) |
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

## Open question

No model is fixed yet for `classify(e)` (axis disambiguation — valid-time vs.
transaction-time, [proposed-model.md](proposed-model.md) §3). Likely folds
into the same adjudication call (gpt-4o-mini) rather than a separate model,
but not yet decided.
