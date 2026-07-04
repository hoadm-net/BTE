# Related Work (surveyed June 2026)

## 1. Conversational / agent memory systems (the baselines)

**Zep / Graphiti** — [Rasmussen et al., *Zep: A Temporal Knowledge Graph
Architecture for Agent Memory*, 2025](https://arxiv.org/abs/2501.13956).
Graphiti's edges already carry a bitemporal model (`valid_at`/`invalid_at` +
ingestion time). On conflict, it closes `invalid_at` on the old edge and adds
the new one — non-destructive. No justification/dependency structure between
edges: invalidation is local to the edge directly contradicted.

**Mem0** — [Chhikara et al., *Mem0: Building Production-Ready AI Agents with
Scalable Long-Term Memory*, 2025](https://arxiv.org/abs/2504.19413).
LLM-mediated `ADD`/`UPDATE`/`DELETE` decision per incoming fact based on
similarity to existing memory. No explicit valid-time/transaction-time
separation, no dependency graph — conflict handling is a per-fact
classification, not a graph traversal.

**Engram** — [*Less Context, More Accuracy: A Bi-Temporal Memory Engine for LLM
Agents*, 2026](https://arxiv.org/abs/2606.09900). Closest existing system to
ours: bitemporal facts (`valid_at`/`invalid_at`, `created_at`/`expired_at`),
non-destructive invalidation with a `supersedes` pointer, and a
cheap-then-escalate conflict pipeline (slot-matching → temporal ordering →
LLM adjudication only on ambiguous cases). The authors name cascading
invalidation of derived facts as future work, not something they solve — their
weakest category (multi-session aggregation, 79.3%) is consistent with this.

**Hindsight** — [Latimer et al., *Hindsight is 20/20: Building Agent Memory that
Retains, Recalls, and Reflects*, 2025](https://arxiv.org/abs/2512.12818).
Splits memory into four networks (World facts, Experience, subjective
Opinions, synthesized Observations). Retrieval fuses four channels — semantic
(embedding cosine), keyword (BM25), graph (spreading activation over entity /
temporal / semantic / causal edges), and temporal (date-range match against an
occurrence interval $[\tau_s, \tau_e]$ plus a mention timestamp $\tau_m$, with
exponential decay $\exp(-\Delta t / \sigma_t)$) — combined via Reciprocal Rank Fusion and a
cross-encoder reranker. Strong reported results: 83.6–91.4% on LongMemEval and
85.7–89.6% on LoCoMo, beating full-context GPT-4o (60.2%) by a wide margin.
Conflict handling exists only for **opinions**: a reinforcement step labels
incoming evidence `{reinforce, weaken, contradict, neutral}` against existing
opinions and adjusts confidence (contradiction halves it, `-2α`). **World
facts** have no equivalent mechanism — background merging resolves conflicts
by recency only, with no notion of derived facts or inference chains. Given
its strong LongMemEval/LoCoMo numbers, a candidate fourth baseline, not just a
retrieval-architecture reference.

**APEX-MEM** — [*APEX-MEM: Agentic Semi-Structured Memory with Temporal
Reasoning for Long-Term Conversational AI*, Amazon Science,
2026](https://arxiv.org/abs/2604.14362). Property graph with domain-agnostic
ontology + append-only storage (nothing overwritten) + a multi-tool retrieval
agent that resolves conflicting or evolving information *at query time*.
Strong results (88.9% LoCoMo QA, 86.2% LongMemEval). A distinct resolution
paradigm: instead of computing and storing a status per edge, it re-resolves
from raw history on every query — avoids storing a wrong status, at the cost
of redoing resolution work every query rather than amortizing it once.

**TiMem** — [*TiMem: Temporal-Hierarchical Memory Consolidation for
Long-Horizon Conversational Agents*, 2026](https://arxiv.org/abs/2601.02845).
Temporal Memory Tree with semantic-guided consolidation across abstraction
levels (75.3% LoCoMo, 76.9% LongMemEval-S). Targets consolidation/abstraction,
not contradiction propagation; no justification structure.

**TSM** — [*Beyond Dialogue Time: Temporal Semantic Memory for Personalized
LLM Agents*, 2026](https://arxiv.org/abs/2601.07468). Builds a semantic
timeline from actual occurrence time rather than dialogue/mention time, and
consolidates temporally continuous information into durative memory (up to
+12.2% over prior methods on LongMemEval/LoCoMo). Relevant to valid-time
reasoning ([bitemporal-model.md](bitemporal-model.md)); no justification
structure for derived facts.

None of the six systems above maintain a justification structure $J$ or
recompute the fixed point $\Phi$ past a directly-contradicted edge — direct
support for [formalism.md](formalism.md)'s Theorem 1 premise.

## 2. Knowledge editing & ripple effects (same phenomenon, different substrate)

This line of work studies the same underlying problem — does correcting one
fact correctly update everything that logically follows from it — but in
**static parametric knowledge edited into LLM weights**, evaluated through
one-shot multi-hop QA rather than a live, append-only graph.

- **RippleEdits** — [Cohen et al., *Evaluating the Ripple Effects of Knowledge
  Editing in Language Models*, TACL 2024](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00644).
  5K diagnostic edits, 6 evaluation criteria (logical generalization,
  compositionality, subject aliasing, preservation, relation specificity).
  Establishes that single-fact edits routinely fail to propagate to logically
  entailed consequences.
- **MQuAKE / MQuAKE-Remastered** — multi-hop QA benchmark where the correct
  answer requires the edited fact's downstream consequences to also be
  updated; the Remastered version addresses evaluation reliability issues in
  the original.
- **ChainEdit** — [*Propagating Ripple Effects in LLM Knowledge Editing through
  Logical Rule-Guided Chains*, ACL 2025](https://arxiv.org/abs/2507.08427).
  Extracts logical rules from a KG and uses them to drive systematic chain
  updates after an edit; +30% logical generalization over prior editing
  methods. Conceptually closest analogue to our propagation step, but the
  rules are pre-defined/extracted from a static KB, not tracked as live
  per-edge justifications, and there is no notion of valid-time vs.
  transaction-time — an edit is just true/false, not bitemporal.
- **RippleCOT** — amplifies ripple-effect propagation via chain-of-thought
  in-context prompting rather than structural dependency tracking.

**Takeaway:** this literature confirms multi-hop propagation failure is a
known, unsolved phenomenon — but always in the knowledge-editing setting
(batch edit to a fixed model/KB), never in an online bitemporal conversational
graph where the propagation must also decide *which time axis* is affected.

## 3. Truth maintenance / incremental materialization (algorithmic foundation)

- **JTMS** — Doyle, *A Truth Maintenance System*, 1979. Single current belief
  state; dependency-directed backtracking on retraction.
- **ATMS** — de Kleer, *An Assumption-Based TMS*, 1986. Multiple simultaneous
  belief environments tagged by assumption sets; nogoods prune inconsistent
  environments without re-deriving them.
- **DRed (Delete-and-Rederive)** — Gupta, Mumick, Subrahmanian, *Maintaining
  Views Incrementally*, SIGMOD 1993. Incremental maintenance of recursive
  Datalog views under negation/aggregation: overestimate-and-delete every fact
  that *might* depend on a deleted base fact, then rederive what's still
  supported by surviving facts.
- **Backward/Forward (BF) and FBF** — Motik et al., *Maintenance of Datalog
  Materialisations Revisited*, and follow-up work. Improves on DRed for facts
  with multiple independent derivations by checking alternative support
  before deleting, avoiding unnecessary rederivation.
- **Provenance semirings** — Green, Karvounarakis & Tannen, *Provenance
  Semirings*, PODS 2007. Generalizes Datalog's least-fixed-point semantics
  from the Boolean truth domain to arbitrary commutative-semiring-valued
  annotations (lineage, confidence, multiplicities), while preserving
  monotonicity and fixed-point existence. This is the framework
  [formalism.md](formalism.md) instantiates with a bitemporal status lattice
  as the annotation domain — the formal grounding for why the soundness/
  termination proofs there are provable rather than asserted.

**Takeaway:** DRed/FBF already solve "what no longer follows when a base fact
is removed" for single-valued (true/false) Datalog views, and provenance
semirings already show fixed-point semantics survive richer value domains.
What's new in [formalism.md](formalism.md) is instantiating that known
generalization with a *specific* bitemporal lattice and proving (Theorem 1)
that the systems surveyed in §1 — none of which maintain any such
fixed-point-tracked status — are structurally incapable of the invariant this
project targets, independent of which benchmark is used to check it.

## 4. Temporal KG reasoning for LLMs (adjacent, not overlapping)

**MemoTime** — [*Memory-Augmented Temporal Knowledge Graph Enhanced LLM
Reasoning*, ACM Web Conf 2026](https://arxiv.org/abs/2510.13614). "Tree of
Time" decomposition enforcing non-decreasing timestamps across a multi-hop
reasoning chain; temporal-first pruning removes paths violating time
constraints before semantic ranking. Addresses *retrieval-time* temporal
consistency for reasoning chains, not contradiction detection or fact
dependency tracking — relevant to our retrieval component, not to propagation.

## 5. Surveys (gap confirmation)

[*Graph-based Agent Memory: Taxonomy, Techniques, and Applications*, 2026](https://arxiv.org/html/2602.05665v1)
covers conflict detection in KG-backed agent memory generally (citing Mem0,
bitemporal invalidation) but explicitly notes that "update propagation
mechanics for cascading changes through fact dependencies are notably absent"
across the systems it surveys — independent confirmation that no surveyed
2025–2026 system does what this project proposes.

## 6. Adjacent benchmarks not adopted

**BeliefShift** ([2026](https://arxiv.org/abs/2603.23848v1)) — tracks belief
consistency and opinion drift across sessions, but the "facts" in question are
subjective user opinions/preferences, not asserted-then-derived factual edges.
Noted for awareness; not used as an evaluation tier (see
[evaluation-benchmarks.md](evaluation-benchmarks.md)).
