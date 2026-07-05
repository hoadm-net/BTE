# Proposed Model

Implements the algorithm formalized in [formalism.md](formalism.md):
maintain Justification-Temporal Soundness (JTS) by computing the least fixed
point of the evaluation operator $\Phi$ over the bitemporal status lattice
$\Sigma$, incrementally, after each contradiction event. Components 1 and 3
implement $J$/$\Sigma$ and $\Phi$/BBP; components 2, 4, 5 are infrastructure
reused from prior systems (cited inline).

## 1. Bitemporal Justification Graph (BJG)

Extends the Graphiti/Engram edge schema with an explicit justification field:

```text
edge {
  subject, relation, object
  t_valid_start, t_valid_end        # world time
  t_transaction                      # ingestion time
  status        ∈ {active, superseded}
  source_type   ∈ {asserted, derived}
  confidence    ∈ [0, 1]
  justification: [edge_id]           # non-empty only if source_type = derived
  supersedes:    edge_id | null       # provenance pointer, per Engram convention
}
```

`justification` is what Zep/Graphiti and Engram lack (confirmed in
[related-work.md](related-work.md) §1) — it is the structural prerequisite for
propagation. A derived edge's justification set can have more than one member
(an edge can be entailed by several independent derivation paths), which
matters for step 3 below.

## 2. Cheap-then-escalate conflict detection

Reused from Engram's design (no need to redesign a part that already works):
slot-matching → temporal-overlap check → LLM adjudication only for ambiguous
cases. Extended in one place: when checking a new edge against the graph, also
check it against the *inherited* valid-time window of derived edges (an
inherited-conflict, per [contradiction-detection.md](contradiction-detection.md)),
not only against asserted edges.

## 3. Bounded Bitemporal Propagation (BBP) — the core algorithm

Adapts DRed (Gupta et al. 1993) and FBF (Motik et al.) from single-valued
Datalog view maintenance to a two-axis bitemporal status. Triggered only when
step 2 confirms a conflict and resolves it (an edge `e` becomes `superseded`).

```text
BBP(e, max_depth, theta):
  axis = classify(e)             # which axis changed: VALID or TRANSACTION
  frontier = { d : e in d.justification }     # depth 1 (DRed: overestimate)
  depth = 1
  while frontier not empty and depth <= max_depth:
    next_frontier = {}
    for d in frontier:
      if axis == VALID:
        d.t_valid_end = min(d.t_valid_end, e.t_valid_end)   # inherit shortened window
      else:  # TRANSACTION
        alt = surviving_justifications(d) - {e}              # FBF-style check
        if alt supports d independently of e:
          continue            # d remains valid, do not cascade further from it
        d.status = superseded
        d.supersedes_reason = e.id
        d.confidence *= decay_factor
      if d.confidence < theta:
        continue               # bounded: stop cascading low-confidence branches
      next_frontier += { d' : d in d'.justification }
    frontier = next_frontier
    depth += 1
```

Two properties not present in classical DRed/FBF, because those operate on a
single true/false axis:

- **`classify(e)`**: decides whether the cascade propagates along the
  valid-time axis (the world changed — shorten inherited windows, edges may
  still be "transaction current") or the transaction-time axis (we were
  wrong — mark superseded regardless of period). This is the axis-disambiguation
  problem flagged in [truth-maintenance-systems.md](truth-maintenance-systems.md)
  as the reason classical TMS doesn't port directly.
- **`alt support` check before marking superseded** (FBF-style, not naive
  DRed): a derived edge with multiple independent justifications survives if
  any one of them still holds — avoids over-invalidating facts that happen to
  share one justification with the contradicted edge but are also entailed
  another way.

### Correctness

This pseudocode implements formalism.md's Theorem 2 and 3:

- With `max_depth = ∞, theta = 0`, this loop *is* the chaotic iteration of
  $\Phi$ used in the Theorem 2 soundness proof — it computes $\sigma^*$
  exactly on the component reachable from `e`. The `alt support` check is
  $\Phi$'s OR-of-ANDs rule for the transaction axis, evaluated faithfully
  (FBF-style, not DRed's overestimate).
- With finite `max_depth` / `theta > 0`, this is the **bounded relaxation**
  characterized in formalism.md §7: a genuine accuracy/cost trade-off, not a
  free optimization. Edges outside the explored frontier are stale-by-omission,
  not provably correct — this is the formal object behind
  [RQ4/H4](../README.md#hypotheses).
- Termination and the `O(D · b)` bound (bounded mode) / `O(m_C)` bound
  (unbounded mode; `m_C` = dependency links in the affected component) are
  formalism.md Theorem 3, not re-derived here.

### Why triggering is selective (cost control, → H4)

BBP runs only as a callback from confirmed-conflict resolution in step 2, never
on the query path. Expected trigger frequency must be measured empirically on
LongMemEval/LoCoMo/BEAM (not assumed low) — see the open item in
[related-work.md](related-work.md) framing and the project README's H4.

## 4. Temporal-validity-aware retrieval

Four-channel hybrid retrieval following Hindsight's pattern — semantic
(embedding), keyword (BM25), graph (spreading activation over entity/temporal/
semantic/causal edges), temporal (date-range match with exponential decay) —
fused via Reciprocal Rank Fusion + cross-encoder rerank. The one addition: every
channel filters on `status = active` and matches the query's reference time
against `t_valid_*`, so retrieval automatically reflects whatever BBP has
propagated — no separate consistency-check step needed at query time. This is
the piece Hindsight's own conflict handling lacks for world facts (it only
reinforces/weakens *opinions*, never tracks dependency between facts — see
[related-work.md](related-work.md) §1).

## 5. Fixed reader

Reader held constant across systems under comparison, following the
fixed-reader protocol used in LongMemEval-V2 (see
[evaluation-benchmarks.md](evaluation-benchmarks.md)) — isolates memory-module
quality from reader quality when comparing against Zep/Mem0/Engram.

## Positioning against prior systems

Evidence for Theorem 1's premise: no surveyed system maintains $\Sigma$ or computes $\Phi$.

| System | Bitemporal edges | Justification tracking | Propagation |
| --- | --- | --- | --- |
| Zep/Graphiti | yes | no | no |
| Mem0 | no | no | no |
| Engram | yes | no (explicit future work) | no |
| APEX-MEM | yes (append-only) | no | no (resolves per-query at retrieval time, not stored) |
| Hindsight | yes (per-fact) | no (opinions get confidence reinforcement only) | no |
| ChainEdit (knowledge editing, different substrate) | no (true/false only) | yes (static KB rules) | yes (batch, one axis) |
| **This work** | yes | yes | yes (online, two axes, bounded — proven, formalism.md) |

## Open implementation questions

- `decay_factor` and `theta`: need an ablation sweep, not a fixed pick (flagged
  already as a risk in the project README).
- `classify(e)` currently assumes the conflict detector already labels an
  invalidation as update-vs-correction (see [bitemporal-model.md](bitemporal-model.md));
  need to confirm this label is reliably inferable from conversational input
  rather than requiring it as metadata.
- Multi-justification derived edges (the `alt support` branch) require derived
  edges to record *all* valid derivation paths, not just the first one found —
  changes the ingestion-time derivation step, not just the propagation step.
