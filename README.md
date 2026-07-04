# Beyond the Edge: Sound and Bounded Contradiction Propagation in Bitemporal Conversational Memory

Conversational memory systems (Zep, Engram, Mem0, ...) resolve contradictions at
the single edge where they are asserted. They do not propagate that invalidation to
facts *derived* from the now-invalid edge, so stale derived facts stay in the graph
across multiple hops.

This repo studies whether propagating contradictions across multi-hop dependency
chains in a bitemporal conversation graph improves long-term QA consistency and
accuracy compared to single-edge resolution.

## Core Theoretical Claim

- **Justification-Temporal Soundness (JTS)**: a memory state is sound iff every
  derived fact's status equals the least fixed point of a monotone evaluation
  operator over its justification graph, computed independently per time axis.
- **Theorem (incompleteness)**: any system that resolves a contradiction only at
  the directly-affected edge — per [docs/related-work.md](docs/related-work.md),
  every surveyed 2025–2026 system — cannot maintain JTS in general, shown by a
  minimal constructive counterexample.
- **Theorem (soundness + termination/complexity)**: the proposed propagation
  algorithm restores JTS over the affected justification subgraph, within a
  characterized, bounded cost.

Full statements and proof sketches are in [docs/formalism.md](docs/formalism.md).
Benchmarks confirm the predicted failure mode occurs in real conversational
data and that the algorithm removes it.

## Research Question

**Main RQ:** Does propagating temporal contradictions across multi-hop dependency
chains in a bitemporal conversation graph improve consistency and accuracy of
long-term QA compared to single-edge contradiction resolution — and by how much?

### Sub-questions

| ID | Focus | Question |
| --- | --- | --- |
| RQ1 | Representation | How can derived facts and their temporal-validity dependencies be formally modeled on a bitemporal graph such that contradictions can propagate? |
| RQ2 | Algorithm | What TMS-inspired algorithm detects and propagates temporal contradictions across multi-hop dependency chains efficiently, while preserving bitemporal consistency? |
| RQ3 | Effectiveness | Does propagation improve performance on contradiction-sensitive and temporal-reasoning questions, and by how much relative to a single-edge baseline? |
| RQ4 | Trade-offs | What is the cost of propagation (latency, tokens, memory), and where does the method sit on the accuracy–latency Pareto frontier? |
| RQ5 | Scope / robustness | How does performance vary with hop depth, edge type (asserted vs. derived), temporal density, and source confidence? |

## Hypotheses

| ID | Statement | Maps to |
| --- | --- | --- |
| H1 | The incompleteness theorem's predicted failure mode is empirically real, not just a logical possibility: SOTA systems (Zep, Engram, Mem0, ...) show a measurable accuracy drop at ≥2 hops that is absent at 1 hop, consistent with never recomputing the fixed point beyond the directly-contradicted edge. | RQ3 |
| H2 | A system that maintains JTS (explicit justification graph + the proven propagation algorithm) tracks the fixed point where single-edge resolution provably cannot, with a measurable accuracy consequence on the diagnostic probe at ≥2 hops. | RQ1, RQ2, RQ3 |
| H3 | The JTS gap is not only a synthetic-probe artifact: it shows up, more modestly, on real-world knowledge-update / temporal-reasoning categories (LongMemEval KU/TR; BEAM CR/EO/KU; STALE) without regressing unrelated categories. | RQ3 |
| H4 | Propagation cost is bounded; because it is selectively triggered (only on contradiction events, not per query), the method does not worsen — and may improve — the accuracy–latency Pareto frontier. | RQ4 |
| H5 | Ablating propagation (single-edge only) or the bitemporal model each produces a measurable, isolable drop on the diagnostic probe. | RQ5 |

## Evaluation Datasets

| Tier | Dataset(s) | Scale |
| --- | --- | --- |
| General | LongMemEval-S, LoCoMo, BEAM-1M | Full (500 / 10 conv., 1986 QA / BEAM-1M) |
| Category-focused | LongMemEval (KU + TR), BEAM (CR + EO + KU), STALE | Category subsets / 400 scenarios |
| Diagnostic | Synthetic multi-hop contradiction-resolution probe | ~150–200 items, generated via templates + LLM-assisted verification, controlled by hop depth (1–4), edge type (asserted/derived), temporal density, source confidence |

Baselines (Zep, Mem0, Engram, ...) are re-run on a shared harness rather than
cited from vendor-reported numbers.
