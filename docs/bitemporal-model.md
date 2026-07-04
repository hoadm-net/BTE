# Bitemporal Data Model

## Two time axes

A fact recorded in a temporal database can be timestamped along two independent
axes (Snodgrass, *Developing Time-Oriented Database Applications*, 1999):

- **Valid time** (`t_valid_start`, `t_valid_end`): the period during which the fact
  was/is true *in the world*, independent of when the system learned about it.
- **Transaction time** (`t_transaction`): the period during which the fact was
  recorded as current *in the database*. Transaction time is append-only — once a
  record is superseded, its old transaction-time interval is closed but never
  rewritten (it remains queryable for "what did we believe at time X").

A model using only one axis is **uni-temporal**. Using both is **bitemporal**.

## Why both axes matter here

The two axes can diverge:

- A fact can be **valid-time correct** but **transaction-time stale** — the system
  learned something and later learned it was wrong, even though the original
  valid-time interval was accurately recorded at the time.
- A fact can be **transaction-time current** but **valid-time expired** — e.g.
  "Alice works at Acme" is still the latest assertion in the DB, but its
  `t_valid_end` has passed (she left last month) and nothing has corrected it yet.
- Corrections and updates are not the same operation. An **update** closes the
  valid-time interval of an old fact and opens a new one (the world changed). A
  **correction** retroactively edits the transaction-time record without
  necessarily implying the world changed (we were wrong about the past).

Contradiction handling needs to know which of these happened, because the
downstream consequences differ: an update invalidates *future* derived facts that
assumed the old value continued to hold; a correction invalidates *all* derived
facts that were ever built on the now-known-wrong assertion, regardless of when
they were derived.

## Bitemporal edges in a conversation graph

Each edge in the conversation graph carries:

```text
edge(subject, relation, object, t_valid_start, t_valid_end, t_transaction)
```

`t_valid_end = null` / "open" means the fact is asserted to still hold. Closing
`t_valid_end` is how an update is represented; marking the edge `superseded` at a
given `t_transaction` (while leaving `t_valid_*` untouched) is how a correction is
represented.

## Asserted vs. derived edges

- **Asserted edge**: directly stated in the conversation (by the user or another
  source).
- **Derived edge**: produced by applying a rule or an LLM inference over one or
  more existing edges. A derived edge's truth is conditional on the truth of the
  edges it was derived from — this dependency is what [truth-maintenance-systems.md](truth-maintenance-systems.md)
  is responsible for tracking.

## Open questions for this project

- Representation of `t_valid_end = null` vs. a derived edge whose dependency has
  closed — does the derived edge inherit an implicit `t_valid_end`, or does it
  require an explicit re-derivation step?
- How temporal overlap is computed between a derived edge's *inherited* validity
  window and a newly asserted conflicting edge (see [contradiction-detection.md](contradiction-detection.md)).
