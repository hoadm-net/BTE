"""Diagnostic probe generator (.plan/04-probe.md; evaluation-benchmarks.md).

Design axes: hop depth 1-4, contradicted edge (asserted root vs derived
conclusion), contradiction axis (update vs correction), temporal density
(adjacent vs distant sessions), source confidence (plain vs hedged).
Full factorial minus the empty depth-1/derived cells, replicated across
three surface domains.

Every item carries two parallel forms plus gold labels:
- sessions: canned natural-language turns (input for real extraction);
- oracle_facts/oracle_retractions: the structured facts an ideal
  extractor would emit (input for LLM-free replay through the Ingestor,
  which is how the generator itself is validated against the formal
  semantics — see tests/test_probe.py).

Gold: pre/post answers for the probed conclusion, the relations expected
superseded after the event, and the event's axis label (for the
classify(e) reliability measurement, H5).
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field

from .rules import ChainRule

DEPTHS = (1, 2, 3, 4)
EDGE_TYPES = ("asserted", "derived")
AXES = ("update", "correction")
DENSITIES = ("adjacent", "distant")
CONFIDENCES = ("high", "low")

FILLERS = [
    "Can you recommend a podcast for my commute?",
    "What's a quick dinner I could cook tonight?",
    "How do people usually get started with journaling?",
    "Any tips for sleeping better on weekends?",
    "What board games work well for two players?",
]


@dataclass
class Domain:
    name: str
    # chain relations: root first, then one per additional hop
    relations: tuple[str, ...]
    conclusions: tuple[str, ...]  # conclusion relation per depth >= 2
    statements: dict[str, str]  # relation -> sentence template
    hedged_root: str
    questions: tuple[str, ...]  # question per depth 1..4
    root_update: str  # replacement or retraction announcement
    root_update_is_retraction: bool
    root_correction: str
    derived_update: tuple[str, ...]  # per depth 2..4
    derived_correction: tuple[str, ...]
    values: dict[str, list]  # pools, keyed by slot name


EMPLOYMENT = Domain(
    name="employment",
    relations=("works_at", "located_in", "city_timezone", "tz_dst_policy"),
    conclusions=("workplace_city", "work_timezone", "work_dst_policy"),
    statements={
        "works_at": "I work at {0}.",
        "located_in": "{0}'s main office is in {1}.",
        "city_timezone": "{0} is in the {1} timezone.",
        "tz_dst_policy": "The {0} timezone {1}.",
    },
    hedged_root="If I remember right, I'm employed at {0}.",
    # hop3/hop4 questions and derived_correction[1] deliberately avoid
    # "work city"/"work timezone" as the grammatical subject - that
    # phrasing reads as a claim about the CITY'S/TIMEZONE'S own
    # geography (which never changes here), not the user's personal
    # effective work-hours timezone (which the derived edge tracks and
    # the correction/update overrides). A reader can default to the
    # geographic (wrong) reading under the old wording even with the
    # correct, unambiguous fact directly retrieved, since both readings
    # are grammatically valid.
    questions=(
        "Which company do I work for?",
        "Which city is my workplace in?",
        "Which timezone do my work hours actually run on?",
        "Does the timezone my work hours run on observe daylight saving time?",
    ),
    root_update="Update: I've left {0} - I'm at {1} now.",
    root_update_is_retraction=False,
    root_correction=("I have to correct myself: I never worked at {0}. "
                     "I've been at {1} all along - I mixed the names up."),
    derived_update=(
        "Heads up - my workplace is in {1} these days, not {0}.",
        "These days my work hours follow the {1} timezone, not {0}.",
        "My work schedule {1} now, it no longer {0}.",
    ),
    derived_correction=(
        "Your notes are wrong - my workplace city is {1}, not {0}.",
        "That timezone was never right: the timezone my work hours "
        "actually run on is {1}, not {0}.",
        "Correction: my work schedule {1}; it never {0}.",
    ),
    values={
        "root": [["Acme Corp", "Apex Analytics"], ["Globex", "Initech"],
                 ["Umbrella Labs", "Stark Industries"]],
        "hop2": ["Denver", "Portland", "Atlanta"],
        "hop3": ["Mountain", "Pacific", "Eastern"],
        "hop4": ["observes daylight saving", "stays on standard time",
                 "observes daylight saving"],
    },
)

RESIDENCE = Domain(
    name="residence",
    relations=("lives_in", "power_utility_of", "billing_portal_of",
               "portal_owner_of"),
    conclusions=("home_utility", "utility_portal", "portal_owner"),
    statements={
        "lives_in": "I live in {0}.",
        # "The electric utility serving {0} is {1}." made {1} (the
        # utility) the sentence's grammatical subject, and extraction
        # matched that grammar over ChainRule's expected direction
        # (subject={0}, the city): it produced (lakeside_power,
        # power_utility_of, Madison) instead of (madison,
        # power_utility_of, Lakeside Power), so derive_closure's
        # subject=mid lookup never matched and no derived edge could be
        # produced. Possessive phrasing, matching employment's
        # "located_in" template, keeps {0} unambiguously the subject.
        "power_utility_of": "{0}'s electric utility is {1}.",
        "billing_portal_of": "{0} handles its billing through {1}.",
        "portal_owner_of": "{0} is owned by {1}.",
    },
    hedged_root="I believe I'm living in {0} at the moment.",
    questions=(
        "Which city do I live in?",
        "Which company supplies my electricity?",
        "Which portal do I pay my power bill on?",
        "Who owns the billing portal I use for power bills?",
    ),
    root_update="Big news - I've moved from {0} to {1}; the relocation is done.",
    root_update_is_retraction=False,
    root_correction=("I misspoke before: I never lived in {0}. "
                     "It's {1}, and it always has been."),
    derived_update=(
        "My electricity supplier switched to {1}, it's not {0} anymore.",
        "I pay my power bill on {1} now instead of {0}.",
        "The portal changed hands - it's owned by {1} now, not {0}.",
    ),
    derived_correction=(
        "Correction: my electricity comes from {1}, never {0}.",
        "Your notes are off - the billing portal is {1}, not {0}.",
        "That was wrong: the portal has always been owned by {1}, not {0}.",
    ),
    values={
        "root": [["Austin", "Boulder"], ["Madison", "Tucson"],
                 ["Raleigh", "Spokane"]],
        "hop2": ["VoltGrid Energy", "Lakeside Power", "Sunbelt Electric"],
        "hop3": ["PayVolt", "GridPay", "WattBill"],
        "hop4": ["Meridian Holdings", "CasCorp", "BluePeak Group"],
    },
)

TRAINING = Domain(
    name="training",
    relations=("training_for", "plan_of", "long_run_day_of",
               "blocked_slot_of"),
    # "long_run_day"/"blocked_slot" previously matched their raw-relation
    # counterpart ("long_run_day_of"/"blocked_slot_of") minus only the
    # "_of" suffix - relation_vocab offers extraction both names as
    # equally valid, and it sometimes picked the conclusion name
    # directly on a third-party fact (e.g. (daniels_2q_plan,
    # long_run_day, ...) instead of (daniels_2q_plan, long_run_day_of,
    # ...)), which derive_closure's exact-relation lookup for r2 never
    # matches, silently producing one fewer derived edge than the chain
    # requires. "training_plan" never collided with "plan_of" this way
    # and was never affected. Prefixed the two colliding conclusions to
    # make them structurally distinct.
    conclusions=("training_plan", "current_long_run_day",
                "current_blocked_slot"),
    # "plan_of"/"blocked_slot_of" previously read as personal statements
    # ("I follow...", "...is off-limits for me") - extraction naturally
    # attributed them DIRECTLY to the user with an explicit premise link
    # (bypassing derive_closure entirely) instead of as a third-party
    # raw fact anchored on {0} (the race / the run day) the way
    # "long_run_day_of" already was. That breaks the NEXT ChainRule in
    # the sequence, which looks up subject=mid by exact relation name,
    # not by premise-chasing, silently missing a derived edge. Rephrased
    # to active voice with {0} as the unambiguous subject, matching
    # "long_run_day_of"'s already-working shape.
    statements={
        "training_for": "I'm training for the {0}.",
        "plan_of": "{0} uses the {1} plan.",
        "long_run_day_of": "The {0} plan schedules long runs on {1}.",
        "blocked_slot_of": "{0} blocks off {1}.",
    },
    hedged_root="I think I've signed up to train for the {0}.",
    questions=(
        "Which race am I training for?",
        "Which training plan am I following?",
        "Which day are my long runs on?",
        "Which time slot should stay free of meetings for me?",
    ),
    root_update="Update: I've withdrawn from the {0} - my knee won't heal in time.",
    root_update_is_retraction=True,
    root_correction=("I got mixed up earlier - I was never signed up for "
                     "the {0}. It's the {1} I'm training for."),
    derived_update=(
        "I've switched plans for the race: {1} now, not {0}.",
        "The plan moved my long runs to {1}, no longer {0}.",
        "With the new schedule, keep {1} free instead of {0}.",
    ),
    derived_correction=(
        "Correction - the plan I follow is {1}, never was {0}.",
        "Your notes are wrong: long runs are on {1}, not {0}.",
        "It was never {0} that's blocked - it's {1}.",
    ),
    values={
        "root": [["Boston Marathon", "Chicago Marathon"],
                 ["Berlin Marathon", "London Marathon"],
                 ["City Half", "Coastal Ultra"]],
        "hop2": ["Pfitzinger 18/55", "Hansons Beginner", "Daniels 2Q"],
        "hop3": ["Saturday mornings", "Sunday mornings", "Friday evenings"],
        "hop4": ["weekend breakfast meetings", "early Saturday calls",
                 "Friday night events"],
    },
)

DOMAINS = (EMPLOYMENT, RESIDENCE, TRAINING)


def domain_rules(d: Domain) -> list[ChainRule]:
    rules = [ChainRule(d.relations[0], d.relations[1], d.conclusions[0])]
    for i in range(1, len(d.conclusions)):
        rules.append(ChainRule(
            d.conclusions[i - 1], d.relations[i + 1], d.conclusions[i]))
    return rules


@dataclass
class ProbeItem:
    probe_id: str
    domain: str
    hop_depth: int
    contradicted: str
    axis: str
    density: str
    confidence: str
    sessions: list[list[str]] = field(default_factory=list)
    oracle_facts: list[list[dict]] = field(default_factory=list)
    oracle_retractions: list[list[dict]] = field(default_factory=list)
    question: str = ""
    gold_pre: str = ""
    gold_post: str = ""
    gold_invalidated: list[str] = field(default_factory=list)
    gold_axis: str = ""


def _fact(subject, relation, obj, confidence=1.0, correction=False) -> dict:
    return {"subject": subject, "relation": relation, "object": obj,
            "valid_from": None, "valid_to": None, "confidence": confidence,
            "is_correction": correction, "premises": []}


def _norm(term: str) -> str:
    from .extraction import normalize
    return normalize(term)


def generate_item(d: Domain, depth: int, contradicted: str, axis: str,
                  density: str, confidence: str, rep: int,
                  rng: random.Random) -> ProbeItem:
    root_pool = d.values["root"][rep % len(d.values["root"])]
    root, root_alt = root_pool
    chain_vals = [root] + [
        d.values[f"hop{i}"][rep % len(d.values[f"hop{i}"])]
        for i in range(2, depth + 1)
    ]

    item = ProbeItem(
        probe_id=f"{d.name[:3]}-d{depth}-{contradicted[:3]}-{axis[:4]}-"
                 f"{density[:3]}-{confidence[:2]}-r{rep}",
        domain=d.name, hop_depth=depth, contradicted=contradicted,
        axis=axis, density=density, confidence=confidence,
        gold_axis=axis,
    )

    # -- statement turns and their oracle facts -------------------------
    hedged = confidence == "low"
    conf_val = 0.6 if hedged else 1.0
    turns: list[tuple[str, dict]] = []
    root_text = (d.hedged_root if hedged else d.statements[d.relations[0]])
    turns.append((root_text.format(root),
                  _fact("user", d.relations[0], root, conf_val)))
    for i in range(2, depth + 1):
        rel = d.relations[i - 1]
        prev, val = chain_vals[i - 2], chain_vals[i - 1]
        turns.append((d.statements[rel].format(prev, val),
                      _fact(_norm(prev), rel, val)))

    # -- contradiction turn ---------------------------------------------
    if contradicted == "asserted":
        if axis == "update":
            text = d.root_update.format(root, root_alt)
            if d.root_update_is_retraction:
                event = ("retract", {"subject": "user",
                                     "relation": d.relations[0],
                                     "object": root, "was_wrong": False})
            else:
                event = ("fact", _fact("user", d.relations[0], root_alt))
        else:
            text = d.root_correction.format(root, root_alt)
            event = ("fact",
                     _fact("user", d.relations[0], root_alt, correction=True))
        # at depth 1 the probed slot is the root itself, so a stated
        # replacement becomes the new answer; deeper conclusions have no
        # context facts for the replacement and go unknown
        has_replacement = not (axis == "update"
                               and d.root_update_is_retraction)
        item.gold_post = (root_alt if depth == 1 and has_replacement
                          else "unknown")
        item.gold_invalidated = [d.relations[0]] + list(
            d.conclusions[:depth - 1])
    else:
        concl_rel = d.conclusions[depth - 2]
        old_val = chain_vals[depth - 1]
        pool = d.values[f"hop{depth}"]
        new_val = pool[(rep + 1) % len(pool)]
        if new_val == old_val:
            new_val = pool[(rep + 2) % len(pool)]
        tmpl = (d.derived_update if axis == "update"
                else d.derived_correction)[depth - 2]
        text = tmpl.format(old_val, new_val)
        event = ("fact", _fact("user", concl_rel, new_val,
                               correction=axis == "correction"))
        item.gold_post = new_val
        item.gold_invalidated = [concl_rel]

    item.question = d.questions[depth - 1]
    item.gold_pre = chain_vals[depth - 1] if depth > 1 else root

    # -- assemble sessions per temporal density -------------------------
    fillers = rng.sample(FILLERS, k=min(3, len(FILLERS)))
    sessions: list[list[str]] = []
    facts: list[list[dict]] = []
    retractions: list[list[dict]] = []

    def push(turn_texts: list[str], turn_facts: list[dict],
             turn_retr: list[dict]) -> None:
        sessions.append(turn_texts)
        facts.append(turn_facts)
        retractions.append(turn_retr)

    if density == "adjacent":
        push([t for t, _ in turns], [f for _, f in turns], [])
    else:
        for i, (t, f) in enumerate(turns):
            push([t], [f], [])
            push([fillers[i % len(fillers)]], [], [])

    if event[0] == "retract":
        push([text], [], [event[1]])
    else:
        push([text], [event[1]], [])
    item.sessions = sessions
    item.oracle_facts = facts
    item.oracle_retractions = retractions
    return item


def generate(seed: int = 20260705, replicates: int = 3) -> list[ProbeItem]:
    rng = random.Random(seed)
    items = []
    for rep in range(replicates):
        d = DOMAINS[rep % len(DOMAINS)]
        for depth in DEPTHS:
            for contradicted in EDGE_TYPES:
                if depth == 1 and contradicted == "derived":
                    continue  # no derived edge exists at depth 1
                for axis in AXES:
                    for density in DENSITIES:
                        for conf in CONFIDENCES:
                            items.append(generate_item(
                                d, depth, contradicted, axis,
                                density, conf, rep, rng))
    return items


def dump(items: list[ProbeItem], path: str) -> None:
    with open(path, "w") as f:
        json.dump([asdict(i) for i in items], f, indent=1)
