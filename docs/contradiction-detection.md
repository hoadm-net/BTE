# Contradiction Detection in a Temporal Knowledge Graph

## What counts as a conflict

Two edges `(s1, r1, o1, valid1)` and `(s2, r2, o2, valid2)` conflict when:

1. **Same subject-relation**: `s1 = s2` and `r1`/`r2` refer to the same relation
   (or a relation known to be functional / single-valued at any instant, e.g.
   "lives in", "works at").
2. **Incompatible objects**: `o1 != o2`, and the relation does not allow multiple
   simultaneous values (a functional-dependency assumption per relation type).
3. **Overlapping valid-time interval**: `valid1` and `valid2` intersect.

If (1) and (2) hold but the valid-time intervals are disjoint, this is not a
contradiction — it is a normal sequence of updates ("worked at A 2019–2021, then
B 2021–present").

## Conflict types

- **Direct conflict**: both edges are asserted, same subject-relation, overlapping
  valid time, incompatible objects. Detected at ingestion time by checking new
  edges against the existing graph.
- **Inherited conflict**: an asserted edge conflicts with the valid-time window a
  *derived* edge inherited from its justification. The derived edge itself was
  never directly contradicted — its premise was.
- **Transitive conflict**: a derived edge conflicts with another derived edge,
  several hops apart in the justification graph, with no direct asserted conflict
  visible at either edge. Only surfaces once propagation is implemented — this is
  the case existing single-edge systems miss.

## Detection vs. propagation

Detection answers "does this new edge conflict with something in the graph right
now?" — a local check against existing edges (functional-dependency violation +
temporal overlap, above). It does not by itself tell you which derived edges
become invalid; that is the propagation step (see
[truth-maintenance-systems.md](truth-maintenance-systems.md)), triggered once a
conflict is detected and one side is resolved (e.g. the newer assertion wins, or
a human/heuristic resolves it).

## Resolution policy (orthogonal to detection)

Common policies once a conflict is confirmed:

- **Recency wins**: the edge with the later `t_transaction` supersedes the other.
- **Confidence-weighted**: the edge with higher source confidence wins regardless
  of recency.
- **Non-destructive supersession**: the losing edge is never deleted, only marked
  `superseded`, with a pointer to what superseded it (provenance chain) — required
  for the bitemporal model since old transaction-time records must stay queryable.

This project assumes non-destructive supersession throughout, since propagation
needs to walk back through superseded edges to find what depended on them.

Direct/inherited/transitive conflict (above) are informal names for any edge
whose $\sigma$ differs from $\sigma^*(G)$, the fixed point of $\Phi$ defined
in [formalism.md](formalism.md). Direct conflicts are caught by checking
asserted edges against the graph; inherited and transitive conflicts require
recomputing $\Phi$ past the directly-affected edge (Theorem 1).
