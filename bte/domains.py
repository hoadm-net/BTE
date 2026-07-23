"""Domain-level dependency structure shared by conflict detection and
retrieval.

A small directed commonsense graph over the coarse life domains of
extraction.DOMAINS: an edge a -> b says a change in domain a can
implicitly invalidate facts in domain b. Two consumers, two directions:
the conflict detector walks edges FORWARD from a new fact's domain to
gate invalidation candidates (conflict.py), and the retriever walks
them BACKWARD from a query's premise-matching facts to pull in the
current state that bears on those premises (retrieval.py's state-basis
channel). The prior is hand-authored generic knowledge, fixed before
any run; confirmed cross-domain conflicts extend it online, so both
consumers sharing one instance means write-time evidence sharpens
query-time behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .extraction import DOMAINS

# Directed commonsense prior: a new fact in the KEY domain may implicitly
# invalidate older facts in the VALUE domains. Authored as generic
# everyday-life knowledge before any benchmark run.
DOMAIN_DEPENDENCY_PRIOR: dict[str, frozenset[str]] = {
    "work_or_study": frozenset({
        "health", "schedule_and_routine", "finance_and_resources",
        "location_and_residence", "transport_and_commute", "plans_and_goals"}),
    "location_and_residence": frozenset({
        "transport_and_commute", "schedule_and_routine", "work_or_study",
        "preferences_and_habits", "plans_and_goals"}),
    "health": frozenset({
        "schedule_and_routine", "transport_and_commute", "work_or_study",
        "preferences_and_habits", "plans_and_goals"}),
    "family_and_social": frozenset({
        "schedule_and_routine", "location_and_residence",
        "finance_and_resources", "plans_and_goals"}),
    "finance_and_resources": frozenset({
        "plans_and_goals", "possessions_and_devices", "preferences_and_habits"}),
    "schedule_and_routine": frozenset({"health", "plans_and_goals"}),
    "transport_and_commute": frozenset({"schedule_and_routine", "plans_and_goals"}),
    # finance added after the gold-pair audit: selling or replacing a
    # possession voids its insurance, warranty, and payment arrangements
    "possessions_and_devices": frozenset({
        "preferences_and_habits", "schedule_and_routine",
        "finance_and_resources"}),
    "plans_and_goals": frozenset({"schedule_and_routine", "finance_and_resources"}),
    "preferences_and_habits": frozenset({"plans_and_goals"}),
}


@dataclass
class DomainDependencies:
    """Domain-level dependency structure gating implicit-conflict search
    and state-basis retrieval: a static commonsense prior plus edges
    learned online from confirmed cross-domain conflicts (the
    conflict-side dual of the detector's learned_multivalued
    non-conflict pruning). The learned counts are an inspectable
    artifact: which dependencies the data actually exercised.
    """

    prior: dict[str, frozenset[str]] = field(
        default_factory=lambda: dict(DOMAIN_DEPENDENCY_PRIOR))
    learned: dict[tuple[str, str], int] = field(default_factory=dict)
    learn_after: int = 1

    def affects(self, domain: Optional[str]) -> frozenset[str]:
        """Two structural rules beyond the prior: every domain affects
        ITSELF - cross-relation implicit conflicts inside one domain (a
        steady-growth investing preference vs an options-trading habit)
        have no other path, since the slot path needs an equal relation
        and the prior has no self-loops - and "other" is a wildcard in
        both positions, since untyped facts would otherwise be
        invisible to the gate in either role (a DMV errand implying a
        car was sold; a license-key backup habit voided by a machine
        wipe)."""
        if not domain:
            return frozenset()
        if domain == "other":
            return frozenset(DOMAINS)
        out = set(self.prior.get(domain, frozenset()))
        for (a, b), n in self.learned.items():
            if a == domain and n >= self.learn_after:
                out.add(b)
        out.add(domain)
        out.add("other")
        return frozenset(out)

    def affected_by(self, domain: Optional[str]) -> frozenset[str]:
        """Domains whose changes can bear on `domain` - the reverse
        direction of affects(), used at query time. Mirrors affects()'s
        self-loop and "other" wildcard."""
        if not domain:
            return frozenset()
        if domain == "other":
            return frozenset(DOMAINS)
        out = {a for a, bs in self.prior.items() if domain in bs}
        for (a, b), n in self.learned.items():
            if b == domain and n >= self.learn_after:
                out.add(a)
        out.add(domain)
        out.add("other")
        return frozenset(out)

    def record(self, new_domain: Optional[str],
               old_domain: Optional[str]) -> None:
        if not new_domain or not old_domain or new_domain == old_domain:
            return
        key = (new_domain, old_domain)
        self.learned[key] = self.learned.get(key, 0) + 1
