# Docs

Background notes on the concepts this project builds on.

- [formalism.md](formalism.md) — **the theoretical core**: status lattice, the JTS invariant, the incompleteness theorem, and the soundness/termination theorems for the propagation algorithm
- [bitemporal-model.md](bitemporal-model.md) — valid time vs. transaction time, bitemporal graphs
- [truth-maintenance-systems.md](truth-maintenance-systems.md) — JTMS / ATMS, justifications, belief revision
- [contradiction-detection.md](contradiction-detection.md) — what counts as a conflict in a temporal knowledge graph
- [evaluation-benchmarks.md](evaluation-benchmarks.md) — LongMemEval, LoCoMo, BEAM, STALE
- [related-work.md](related-work.md) — literature survey: baselines (Zep, Mem0, Engram, Hindsight, APEX-MEM, ...), knowledge editing/ripple effects, TMS/incremental view maintenance, temporal KG reasoning
- [proposed-model.md](proposed-model.md) — the architecture implementing the algorithm in formalism.md
- [models.md](models.md) — concrete model choice per pipeline role (extraction, adjudication, reader, judge, embedding), with current pricing and the GPT-5 mini/nano cost comparison
