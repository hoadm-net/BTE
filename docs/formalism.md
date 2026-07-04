# Formal Model: Bitemporal Status Lattice and Propagation Guarantees

This is the theoretical core of the project. Everything else (architecture,
algorithm pseudocode, benchmarks) is either an implementation of, or empirical
evidence for, the claims made here. The structure is: one invariant, one
incompleteness theorem (proof, not benchmark) showing every surveyed system
violates it, and two theorems showing the proposed algorithm restores it
within a characterized cost.

## 1. Setup

Graph $G = (V, E)$, $E = E_{\text{asserted}} \sqcup E_{\text{derived}}$. Every
edge $e$ has a justification $J(e)$, a set of *alternative* derivation sets
($J(e) = \emptyset$ for asserted edges — they are not derived from anything).
An edge $e \in E_{\text{derived}}$ is justified if at least one
$J_i \in J(e)$ holds, where "holds" means every member of $J_i$ holds. $J$ is
required to form a DAG (no edge participates in its own justification,
transitively) — enforced at insertion time, not assumed for free.

## 2. Status domain

Per axis (valid-time, transaction-time), define a 3-element status domain:

$$
\mathbb{S} = \{\bot, U, \top\}, \qquad \bot \sqsubseteq U \sqsubseteq \top
$$

$\top$ = known active on this axis, $\bot$ = known invalid, $U$ = not yet
determined. This is an *information* order (how much is known), not a truth
order — standard in 3-valued fixed-point semantics (Kleene; later used in
well-founded semantics for logic programs with negation, Van Gelder/Ross/
Schlipf 1991).

Joint status:

$$
\Sigma = \mathbb{S}_{\text{valid}} \times \mathbb{S}_{\text{trans}}
$$

ordered componentwise. $\Sigma$ is a finite lattice (9 elements, bounded
height) with bottom $(\bot,\bot)$ and top $(\top,\top)$. A **status
assignment** is a function $\sigma : E \to \Sigma$.

## 3. Evaluation operator $\Phi$

For asserted edges, $\sigma(e)$ is given (input). For a derived edge $d$ with
justification alternatives $J(d) = \{J_1, J_2, \dots\}$:

$$
\sigma_{\text{valid}}(d) =
\begin{cases}
\top & \text{if } \exists\, J_i \in J(d) : \forall e \in J_i,\ \sigma_{\text{valid}}(e) = \top \quad \text{(over the intersected valid window)} \\[4pt]
\bot & \text{if } \forall\, J_i \in J(d) : \exists\, e \in J_i,\ \sigma_{\text{valid}}(e) = \bot \\[4pt]
U & \text{otherwise}
\end{cases}
$$

(symmetric for $\sigma_{\text{trans}}$). This is an OR-of-ANDs over a finite
domain — a **negation-free (positive) formula**, hence **monotone** with
respect to $\sqsubseteq$ on $\Sigma$: more information in, never less
information out. This single property is what makes everything below
provable rather than asserted.

## 4. Theorem 0 — fixed-point existence (free, from Knaster–Tarski)

$(E \to \Sigma)$ ordered pointwise is a finite, hence complete, lattice;
$\Phi$ is monotone on it. By Knaster–Tarski, $\Phi$ has a unique least fixed
point $\sigma^*(G)$, reachable by iterating $\Phi$ from the all-$U$
assignment. This is the same argument underlying classical Datalog
least-model semantics, generalized from the Boolean truth domain to a richer
annotation domain — an instance of the **provenance semiring** framework
(Green, Karvounarakis & Tannen, *Provenance Semirings*, PODS 2007), with
$\Sigma$ as a semiring tailored to bitemporal status rather than lineage or
confidence.

**Definition — Justification-Temporal Soundness (JTS):** a memory state
$\sigma$ is JTS-sound iff $\sigma = \sigma^*(G)$. This is the consistency
notion the rest of the project is about restoring after a contradiction.

## 5. Theorem 1 — Incompleteness of single-edge resolution

**Claim.** Any procedure $R$ that, on ingesting an edge $e'$ contradicting
$e$, updates only $\sigma(e)$ (and inserts $e'$) and leaves $\sigma(d)$
unchanged for every $d \neq e$, does not compute $\sigma^*(G)$ in general.

**Proof (constructive, minimal counterexample).** Let
$G = \{A \ (\text{asserted}),\ B \ (\text{derived},\ J(B) = \{\{A\}\})\}$,
with $\sigma(A) = (\top,\top)$. Then $\sigma^*(B) = (\top,\top)$. Now ingest
$A'$, a transaction-time correction of $A$ (we were wrong, not the world
changed): $R$ sets $\sigma(A) = (\top,\bot)$ and leaves
$\sigma(B) = (\top,\top)$ untouched. Recomputing $\Phi$ with the updated
$\sigma(A)$ gives $\sigma_{\text{trans}}(B) = \bot$ (its sole justification is
now transaction-invalid), so
$\sigma^*(B) = (\top,\bot) \neq (\top,\top) = \sigma(B)$ under $R$. $R$'s
output is not the fixed point. $\blacksquare$

This is **data-independent** — it holds for any system fitting the
"single-edge" description, which per [related-work.md](related-work.md) §1
includes Zep/Graphiti, Mem0, Engram, Hindsight (for world facts), APEX-MEM
(resolves at query time, never recomputes $\Phi$ over stored status), TiMem,
and TSM. None of them maintain $\Sigma$ or a justification structure $J$, so
none can compute $\sigma^*$ — this is a structural gap, provable without
running a single benchmark. The diagnostic probe and STALE/BEAM-CR results
([evaluation-benchmarks.md](evaluation-benchmarks.md)) exist to confirm this
predicted failure mode actually occurs in realistic conversational data, not
to establish it.

## 6. Theorem 2 — Soundness of unbounded BBP

**Claim.** $\text{BBP}(e, \texttt{max\_depth}=\infty, \texttt{theta}=0)$
computes $\sigma^*$ exactly, restricted to the connected justification
component reachable from $e$.

**Proof sketch.** Each round of BBP's frontier expansion is one application
of $\Phi$ restricted to the edges depending on the changed edge. Since $\Phi$
is monotone on a finite-height lattice, chaotic iteration of $\Phi$ over the
affected component converges to its least fixed point in finitely many
rounds (standard argument for monotone operators on finite lattices; the same
correctness argument used for DRed/FBF, generalized from the Boolean lattice
to $\Sigma$). The alt-support check in BBP's transaction-axis branch is
exactly the OR-of-ANDs rule for $\sigma_{\text{trans}}$ evaluated faithfully
(this is why BBP follows FBF's alternative-support checking rather than
DRed's delete-then-rederive overestimate, which would be sound but wasteful
here).

## 7. Theorem 3 — Termination (unbounded) and the bounded relaxation

**Unbounded case.** Termination is guaranteed even with
$\texttt{max\_depth} = \infty$, because $J$ is a finite DAG — no infinite
justification chains exist by construction (§1). Cost: $O(|C|)$ where $C$ is
the connected component of derived edges reachable from $e$.

**Bounded case** ($\texttt{max\_depth} = D$, $\texttt{theta} > 0$). This is a
genuine **relaxation**, not a free optimization — it must be stated honestly
rather than oversold. Bounding the frontier at depth $D$ or pruning branches
below confidence threshold $\theta$ means edges outside the explored frontier
keep their prior status even when $\Phi$ would have changed it; they become
**stale by omission** until a later propagation event reaches them (e.g. the
next contradiction that touches their component) or a query forces
on-demand re-evaluation. The cost bound for this relaxed mode is
$O(D \cdot b)$ ($b$ = per-hop branching factor), independent of $|E|$ — this
is the formal object behind [RQ4/H4](../README.md#hypotheses): the
accuracy–latency Pareto trade-off is literally the trade-off between
$(D, \theta)$ and the size of the stale-by-omission set, not a vague
engineering knob.

## 8. Relationship to existing algorithmic families

- **DRed/FBF** (Gupta et al. 1993; Motik et al.) are the special case of this
  framework where $\Sigma$ degenerates to the 2-element Boolean lattice
  $\{\bot, \top\}$ — single axis, no $U$, no bitemporal split. The lift from
  Boolean to the bitemporal product lattice is what turns "which axis does
  this invalidation propagate along" from an ad hoc implementation decision
  (as in [truth-maintenance-systems.md](truth-maintenance-systems.md)'s
  informal discussion) into a first-class value the operator $\Phi$ computes.
- **Provenance semirings** (Green et al. 2007) is the general framework this
  instantiates; no surveyed 2025–2026 system ([related-work.md](related-work.md))
  formalizes contradiction propagation this way — this instantiation appears
  to be new.
- **ATMS nogoods** (de Kleer 1986): the alt-support check is structurally an
  ATMS-style "does some environment avoiding the retracted assumption still
  support this belief" check, specialized to per-axis truth instead of full
  assumption-set tracking — cheaper than full ATMS because $\Sigma$'s bounded
  height avoids ATMS's combinatorial environment-lattice blowup.

## 9. Honest limitations of the formal model

- The model assumes $\texttt{classify}(e)$ (which axis an invalidation
  propagates along) is given correctly. Misclassification — e.g.
  conversational ambiguity about whether something is an update or a
  correction — is **outside** this formal model; it is an NLU/extraction
  problem, to be measured empirically via the diagnostic probe
  ([RQ5/H5](../README.md#hypotheses)), not something the propagation
  algorithm can be proven correct against.
- Confidence ($\texttt{confidence} \in [0,1]$) and the threshold $\theta$ are
  layered on top of $\Sigma$ as a scalar, not integrated into the lattice
  itself. A cleaner treatment might fold confidence into a richer semiring
  (e.g. probabilistic or tropical) instead of bolting it on as a side
  channel — flagged as a possible extension, not solved here.
- All proofs above are sketches at the rigor level appropriate for a
  planning document. Tightening them (full induction, explicit
  lattice-height constants, a precise statement of "stale-by-omission" as a
  formal approximation-error bound) is required before they could appear in
  a paper draft as stated.
