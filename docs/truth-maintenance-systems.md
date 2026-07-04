# Truth Maintenance Systems (TMS)

A TMS tracks *why* a system believes each fact, so that when a belief is retracted,
everything that depended on it can be found and re-evaluated. Originates from
Doyle, "A Truth Maintenance System" (1979) and de Kleer, "An Assumption-based TMS"
(1986).

## Justifications

Every non-primitive belief has one or more **justifications**:

```text
justification(fact, [in-list], [out-list]) -> believed | disbelieved
```

`fact` is believed if at least one justification holds: every fact in its
`in-list` is currently believed and every fact in its `out-list` is currently
disbelieved. A fact with no surviving justification becomes disbelieved
(**out**), not deleted — the record stays, only its status flips.

In this project's terms: a *derived edge* is a TMS-style belief whose
justification is the set of edges (asserted or derived) it was inferred from.

## JTMS vs. ATMS

- **JTMS (Justification-based TMS)**: maintains a single current belief state.
  Retracting a fact triggers **dependency-directed backtracking** — walk the
  justification graph forward from the retracted fact, find every belief whose
  *every* justification is now broken, mark it out, and recurse. Cheap, but only
  ever represents one consistent world view at a time.
- **ATMS (Assumption-based TMS)**: tags every belief with the *set of underlying
  assumptions* it depends on, and can hold multiple, possibly mutually
  inconsistent, belief sets (**environments**) simultaneously. More expensive, but
  avoids redoing the same inference when the system needs to reason about
  alternative pasts (e.g. "what if Alice's stated employer was wrong").

A **nogood** is a set of assumptions known to be jointly inconsistent — once
discovered, any environment that is a superset of a nogood is pruned without
re-deriving the contradiction.

## Why this doesn't port directly to a bitemporal graph

Classical TMS believes/disbelieves along a single axis. Here, a belief's status
depends on *two* independent axes (valid time, transaction time — see
[bitemporal-model.md](bitemporal-model.md)). Dependency-directed backtracking has
to decide, for each dependent edge, whether the retraction invalidates it on the
valid-time axis (the world changed, so anything inferred for the affected period
is wrong), the transaction-time axis (we were wrong, so anything ever inferred
from this assertion is wrong, regardless of period), or both. This is the core
algorithmic problem for [RQ2](../README.md#research-question).

## Formalization

The believed/disbelieved, in-list/out-list description above is a 2-valued
instance of the status lattice $\Sigma$ in [formalism.md](formalism.md):
justifications become the operator $\Phi$, and soundness/termination/complexity
are proven there as Theorem 2 and 3, with Theorem 1 showing why no
single-edge system can satisfy them.

## Relevant prior art outside classical AI

- Datalog with negation / stratified semantics: similar "what no longer follows"
  problem for derived facts in logic-based databases.
- Incremental view maintenance in databases: recomputing materialized views when
  base tables change is structurally the same problem as recomputing derived
  edges when base edges change.
