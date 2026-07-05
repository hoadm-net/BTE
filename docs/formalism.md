# Formal Model: Bitemporal Status Lattice and Propagation Guarantees

This is the theoretical core of the project. Everything else (architecture,
algorithm pseudocode, benchmarks) is either an implementation of, or empirical
evidence for, the claims made here. The structure is: one invariant, one
incompleteness theorem (proof, not benchmark) showing every surveyed system
violates it, and two theorems showing the proposed algorithm restores it
within a characterized cost. Proofs below are full at planning-doc rigor
(hardened 07/2026 from earlier sketches); remaining gaps are listed in §9.

## 1. Setup

Graph $G = (V, E)$, $E = E_{\text{asserted}} \sqcup E_{\text{derived}}$. Every
edge $e$ has a justification $J(e)$, a set of *alternative* derivation sets
($J(e) = \emptyset$ for asserted edges — they are not derived from anything).
An edge $e \in E_{\text{derived}}$ is justified if at least one
$J_i \in J(e)$ holds, where "holds" means every member of $J_i$ holds.

Write $u \to d$ when $u \in \bigcup J(d)$ (u is a justification member of d);
this **dependency relation** is required to be acyclic — no edge participates
in its own justification, transitively. Acyclicity is enforced at insertion
time by an ancestor check (cost: one reverse traversal, $O(|\text{anc}(d)|)$
per inserted derived edge), not assumed for free. Define the **rank** of an
edge: $r(e) = 0$ for asserted edges, $r(d) = 1 + \max \{ r(u) : u \to d \}$
for derived edges — finite and well-defined exactly because the dependency
relation is a finite DAG. Let $R = \max_d r(d)$.

## 2. Status domain

Per axis (valid-time, transaction-time), a 3-element chain:

$$
\mathbb{S} = \{\bot, U, \top\}, \qquad \bot \sqsubseteq U \sqsubseteq \top
$$

$\top$ = known active on this axis, $\bot$ = known invalid, $U$ = not
determined. This is the Kleene **truth order** on three values (false <
unknown < true). Corrected from an earlier draft, which called it an
information order and cited well-founded semantics: the information order of
3-valued fixed-point semantics places $U$ *below* both $\bot$ and $\top$, and
the well-founded machinery exists to handle negation. Neither applies here —
the operator in §3 is negation-free, so monotonicity in the truth order is
all that is needed, and (by Theorem 0(b)) the fixed point is unique anyway,
so nothing hinges on which order supplies the existence argument.

Joint status:

$$
\Sigma = \mathbb{S}_{\text{valid}} \times \mathbb{S}_{\text{trans}}
$$

ordered componentwise: a finite distributive lattice, 9 elements, height 4
(each coordinate contributes a chain of height 2), bottom $(\bot,\bot)$, top
$(\top,\top)$. A **status assignment** is a function $\sigma : E \to \Sigma$.
Meet and join are componentwise min and max on the chains.

A status is read *relative to the edge's recorded bitemporal window*:
$\sigma_{\text{valid}}(e) = \top$ means "active over the window
$[t_{\text{valid\_start}}, t_{\text{valid\_end}})$ as recorded". Interval
arithmetic (shortening a window rather than annulling it) is a refinement
below the granularity of $\Sigma$; see §9.

## 3. Evaluation operator $\Phi$

For asserted edges, $\sigma(e)$ is given (input). For a derived edge $d$ with
$J(d) = \{J_1, J_2, \dots\}$, per axis $\alpha \in \{\text{valid},
\text{trans}\}$:

$$
\Phi(\sigma)_\alpha(d) \;=\; \bigvee_{J_i \in J(d)} \; \bigwedge_{e \in J_i} \sigma_\alpha(e)
$$

— join over alternatives of the meet within each alternative (OR-of-ANDs).
Unfolding join/max and meet/min on the chain recovers the case form: $\top$
if some alternative is all-$\top$; $\bot$ if every alternative contains a
$\bot$; $U$ otherwise. (For the valid axis, the all-$\top$ case is
additionally read over the intersected valid window — the interval
refinement of §9.)

The formula is **negation-free (positive)**, hence $\Phi$ is **monotone**
with respect to $\sqsubseteq$: joins and meets are monotone, compositions of
monotone maps are monotone. Moreover $(\Sigma, \vee, \wedge, (\bot,\bot),
(\top,\top))$ is a commutative idempotent semiring (any bounded distributive
lattice is), so $\Phi$ is literally a provenance-semiring evaluation (Green,
Karvounarakis & Tannen, *Provenance Semirings*, PODS 2007) with $\Sigma$ as
the annotation domain. This single monotonicity property is what makes
everything below provable rather than asserted.

## 4. Theorem 0 — fixed-point existence and uniqueness

**Claim.** (a) $\Phi$ has a least fixed point $\sigma^*(G)$ extending the
asserted inputs. (b) This fixed point is the *only* fixed point extending
those inputs, and synchronous iteration of $\Phi$ from **any** initial
assignment reaches it in at most $R$ rounds ($R$ = maximum rank, §1).

**Proof.** (a) The assignment space $(E_{\text{derived}} \to \Sigma)$ ordered
pointwise is a finite, hence complete, lattice; $\Phi$ is monotone on it
(§3). Knaster–Tarski gives a least fixed point; Kleene iteration from the
all-$(\bot,\bot)$ assignment reaches it, with at most $4\,|E_{\text{derived}}|$
strict increases in total (pointwise lattice height = per-edge height 4 times
the number of edges).

(b) Uniqueness, by strong induction on rank. Any fixed point $\sigma$
satisfies $\sigma(d) = \Phi(\sigma)(d)$, and $\Phi(\sigma)(d)$ depends only
on values of edges $u \to d$, all of rank $< r(d)$. Rank-0 values are the
asserted inputs, shared by every fixed point. If two fixed points agree on
all edges of rank $< k$, the equation forces agreement at rank $k$. So all
fixed points coincide. For convergence from an arbitrary start: after round
$t$ of synchronous application, every edge of rank $\le t$ has its (unique)
fixed-point value, by the same induction; hence $R$ rounds suffice. $\square$

This is the classical Datalog least-model argument generalized from the
Boolean truth domain to the semiring $\Sigma$; uniqueness is the extra fact
bought by requiring $J$ to be a DAG (recursive justification cycles are what
would separate least from greatest fixed points, and they are excluded at
insertion time).

**Definition — Justification-Temporal Soundness (JTS):** a memory state
$\sigma$ is JTS-sound iff $\sigma = \sigma^*(G)$. This is the consistency
notion the rest of the project is about restoring after a contradiction.

## 5. Theorem 1 — Incompleteness of single-edge resolution

**Definition (single-edge resolver).** A procedure $R$ that, on ingesting an
edge $e'$ contradicting $e$ in state $(G, \sigma)$, outputs
$(G \cup \{e'\}, \sigma')$ with $\sigma'(x) = \sigma(x)$ for every
$x \notin \{e, e'\}$ — it may update the contradicted edge and insert the
new one, but touches nothing else.

**Claim.** For every single-edge resolver $R$ there is a two-edge input on
which $\sigma' \neq \sigma^*(G \cup \{e'\})$, on either axis.

**Proof (constructive, minimal counterexample, parameterized by axis).** Let
$\alpha \in \{\text{valid}, \text{trans}\}$ be the axis the contradiction
affects. Let $G = \{A \ (\text{asserted}),\ B \ (\text{derived},\ J(B) =
\{\{A\}\})\}$ with $\sigma(A) = (\top,\top)$; by Theorem 0,
$\sigma^*(B) = (\top,\top)$, and assume the state is JTS-sound before the
event. Ingest $A'$ invalidating $A$ on axis $\alpha$: for
$\alpha = \text{trans}$, a correction (we were wrong — the record was never
right); for $\alpha = \text{valid}$, an annulment of $A$'s recorded window
(the world is not as recorded over that window). $R$ lowers coordinate
$\alpha$ of $\sigma'(A)$ to $\bot$ and, by definition, leaves
$\sigma'(B) = (\top,\top)$. But $B$'s sole justification alternative now
contains a $\bot$ on axis $\alpha$, so $\Phi$ forces
$\sigma^*_\alpha(B) = \bot$: $\sigma'(B) \neq \sigma^*(B)$. $\square$

Remark: an *update* that merely shortens $A$'s window (rather than annulling
it) is the interval-refined version of the $\alpha = \text{valid}$ case —
$B$'s inherited window must shorten correspondingly, which a single-edge
resolver equally cannot do. BBP's `min()` on inherited windows
([proposed-model.md](proposed-model.md) §3) is that refinement.

The counterexample is **data-independent** — two edges, one derivation, no
exotic structure — so it applies to any system fitting the single-edge
definition, which per [related-work.md](related-work.md) §1 includes
Zep/Graphiti, Mem0, Engram, Hindsight (for world facts), APEX-MEM (resolves
at query time, never recomputes stored status), TiMem, and TSM. None of them
maintain $\Sigma$ or a justification structure $J$, so none can compute
$\sigma^*$ — a structural gap, provable without running a single benchmark.
The diagnostic probe and STALE/BEAM-CR results
([evaluation-benchmarks.md](evaluation-benchmarks.md)) exist to confirm this
predicted failure mode occurs in realistic conversational data, not to
establish it.

## 6. Theorem 2 — Soundness of unbounded BBP

**Claim.** After a contradiction event lowers the status of an edge $e$,
$\text{BBP}(e, \texttt{max\_depth}=\infty, \texttt{theta}=0)$ terminates
(Theorem 3) with $\sigma = \sigma^*(G')$ on all of $E$ — exactly on the
dependency descendants of $e$ (the affected component $C$), and trivially
elsewhere.

**Proof.** Write $\sigma_{\text{old}} = \sigma^*(G)$ for the pre-event state
(JTS-sound by induction over events) and $\sigma^* = \sigma^*(G')$ for the
post-event fixed point.

*Outside $C$.* If $d \notin C$, no justification ancestor of $d$ changed, so
the rank induction of Theorem 0(b) run on $G'$ gives
$\sigma^*(d) = \sigma_{\text{old}}(d)$; BBP never visits $d$, so its value is
already correct.

*Inside $C$.* BBP is chaotic iteration of $\Phi$ restricted to $C$:
each frontier processing of a derived edge $d$ recomputes $d$'s status from
the current values of its justification members (the alt-support check *is*
the join-of-meets $\bigvee_i \bigwedge_{e \in J_i}$ on the transaction axis,
evaluated faithfully — FBF-style alternative-support checking rather than
DRed's delete-then-rederive overestimate, which would be sound but wasteful
here); and whenever $d$'s status strictly changes, all dependents of $d$ are
re-enqueued (the `next_frontier` rule). Two facts:

1. *All updates descend.* The event lowered $\sigma(e)$, so
   $\Phi_{G'}(\sigma_{\text{old}}) \sqsubseteq \sigma_{\text{old}}$
   pointwise ($\Phi$ monotone, inputs only went down): $\sigma_{\text{old}}$
   is a post-fixpoint of the new operator, and every recomputation can only
   lower values. Values live in a finite-height lattice, so only finitely
   many strict drops occur (Theorem 3).
2. *Termination state is a fixed point.* At termination the frontier is
   empty, so for every $d \in C$: $d$'s last recomputation set
   $\sigma(d) = \Phi(\sigma)(d)$ using member values that have not changed
   since (any later member change would have re-enqueued $d$). Hence
   $\sigma = \Phi(\sigma)$ on $C$; combined with the unchanged, still-fixed
   values outside $C$, the terminal $\sigma$ is a fixed point of $\Phi_{G'}$
   extending the asserted inputs. By Theorem 0(b) it equals $\sigma^*$.
   $\square$

The valid-axis branch of the pseudocode (window `min()`) is the interval
refinement of the same recomputation; at lattice granularity it is the
$\alpha = \text{valid}$ instance of the join-of-meets rule (§9 for the
formalization gap).

## 7. Theorem 3 — Termination and cost; the bounded relaxation

**Unbounded case.** Let $m_C$ be the number of dependency links
$u \to d$ inside the affected component $C$. Each edge is (re-)enqueued only
when one of its justification members strictly drops; each member's status
can strictly drop at most 4 times ($\Sigma$ has height 4). So $d$ is
processed at most $4 \cdot |\{u : u \to d\}| + 1$ times, and total work is
$O(m_C)$ edge-recomputations — $O(|C| \cdot \bar{b})$ for average in-degree
$\bar{b}$, refining the $O(|C|)$ stated in earlier drafts. Termination is
immediate: finitely many strict drops, and no enqueue without one.
(Precondition: $J$ acyclic — enforced at insertion, §1.)

**Bounded case** ($\texttt{max\_depth} = D$, $\texttt{theta} > 0$). A genuine
**relaxation**, not a free optimization. Formally: call a dependency link
$u \to d$ **cut** if $u$'s status changed during the run but $d$ was not
re-enqueued afterward (because $u$ sat at depth $D$, or $u$'s confidence fell
below $\theta$); let $K$ be the set of edges with a cut incoming link.

**Proposition (stale-by-omission).** At termination of bounded BBP, the
stale set $S = \{d : \sigma(d) \neq \sigma^*(d)\}$ satisfies
$S \subseteq K \cup \text{Desc}(K)$ (dependency descendants of cut points).

*Proof.* Contrapositive, by rank induction inside $C$: if $d \notin K \cup
\text{Desc}(K)$, then no evaluation due at $d$ or any of its ancestors was
skipped, so the Theorem 2 argument applies verbatim to the sub-DAG of $d$'s
ancestors and gives $\sigma(d) = \sigma^*(d)$. $\square$

Corollary: $|S| \le \sum_{c \in K} (1 + |\text{Desc}(c)|)$ — the
accuracy–cost trade-off of [RQ4/H4](../README.md#hypotheses) is literally
the trade-off between $(D, \theta)$ and this bound, not a vague engineering
knob. Stale edges remain stale until a later propagation event's explored
region covers them, or a query triggers **on-demand exact re-evaluation**:
recursively evaluating $\Phi$ over $d$'s ancestors (memoized) returns
$\sigma^*(d)$ in $O(|\text{anc}(d)|)$ — the fallback that makes bounded mode
safe to expose at query time. Frontier cost is $O(D \cdot b)$ ($b$ =
per-hop branching factor), independent of $|E|$.

## 8. Relationship to existing algorithmic families

- **DRed / B/F / FBF** (Gupta, Mumick & Subrahmanian, SIGMOD 1993; Motik,
  Nenov, Piro & Horrocks — B/F, AAAI 2015; FBF in *Maintenance of Datalog
  Materialisations Revisited*, Artificial Intelligence 2019; citations
  checked 2026-07-05) are the special case of this framework where $\Sigma$
  degenerates to the Boolean lattice $\{\bot, \top\}$ — single axis, no $U$,
  no bitemporal split. The lift from Boolean to the bitemporal product
  lattice is what turns "which axis does this invalidation propagate along"
  from an ad hoc implementation decision (as in
  [truth-maintenance-systems.md](truth-maintenance-systems.md)'s informal
  discussion) into a first-class value the operator $\Phi$ computes.
- **Provenance semirings** (Green et al. 2007) is the general framework this
  instantiates — literally, since $(\Sigma, \vee, \wedge)$ is a distributive-
  lattice semiring (§3). No surveyed 2025–2026 system
  ([related-work.md](related-work.md)) formalizes contradiction propagation
  this way; this instantiation appears to be new.
- **ATMS nogoods** (de Kleer 1986): the alt-support check is structurally an
  ATMS-style "does some environment avoiding the retracted assumption still
  support this belief" check, specialized to per-axis truth instead of full
  assumption-set tracking — cheaper than full ATMS because $\Sigma$'s bounded
  height avoids ATMS's combinatorial environment-lattice blowup.

## 9. Honest limitations of the formal model

- **Axis classification is an input.** The model assumes
  $\texttt{classify}(e)$ (which axis an invalidation propagates along) is
  given correctly. Misclassification — conversational ambiguity about
  whether something is an update or a correction — is outside the formal
  model; it is an NLU/extraction problem, measured empirically via the
  diagnostic probe ([RQ5/H5](../README.md#hypotheses)), not something the
  propagation algorithm can be proven correct against. What is proven:
  propagation is sound *given* the label.
- **Interval granularity.** $\Sigma$ tracks status relative to each edge's
  recorded window; window *shortening* (an update that truncates rather than
  annuls) lives below this granularity. BBP handles it operationally
  (`min()` on inherited windows) and Theorem 1's remark covers it
  informally, but a fully interval-indexed status domain (status as a
  function of reference time) is not formalized here. This is the one
  remaining formal gap; candidate fix is a status function
  $\sigma : E \to (\mathbb{T} \to \Sigma)$ with pointwise order, which
  preserves monotonicity but complicates the height/cost constants.
- **Confidence stays a side channel** (decision recorded 07/2026):
  $\texttt{confidence} \in [0,1]$ and the threshold $\theta$ are layered on
  top of $\Sigma$ as a scalar, not folded into the lattice. A richer semiring
  (probabilistic or tropical) could integrate it; deferred as future work,
  stated as a limitation in the paper, not solved here.
- **Rigor level.** Theorems 0–3 above are full proofs at planning-doc rigor
  (hardened from the earlier sketches, 07/2026), but have not yet passed a
  second reader; the §8 special-case claims (DRed/FBF as the Boolean
  degeneration) are verified against the papers' published descriptions,
  with a full read of Motik et al. 2019 still pending before the paper's
  related-work section asserts them.
