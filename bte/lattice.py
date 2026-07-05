"""Bitemporal status lattice and evaluation rule (formalism.md, sections 2-3).

Per-axis domain S is the three-element chain BOT < U < TOP under Kleene's
truth order. The joint status Sigma is the componentwise product; the
evaluation rule for a derived edge is join-over-alternatives of
meet-within-alternative (OR-of-ANDs), which is monotone because it is
negation-free.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Iterable, NamedTuple


class S(IntEnum):
    BOT = 0
    U = 1
    TOP = 2


class Sigma(NamedTuple):
    valid: S
    trans: S


BOTTOM = Sigma(S.BOT, S.BOT)
UNKNOWN = Sigma(S.U, S.U)
TOP = Sigma(S.TOP, S.TOP)

ALL_SIGMA = tuple(Sigma(v, t) for v in S for t in S)


def leq(a: Sigma, b: Sigma) -> bool:
    """Componentwise order; tuple comparison would be lexicographic."""
    return a.valid <= b.valid and a.trans <= b.trans


def meet(a: Sigma, b: Sigma) -> Sigma:
    return Sigma(S(min(a.valid, b.valid)), S(min(a.trans, b.trans)))


def join(a: Sigma, b: Sigma) -> Sigma:
    return Sigma(S(max(a.valid, b.valid)), S(max(a.trans, b.trans)))


def evaluate(alternatives: Iterable[Iterable[Sigma]]) -> Sigma:
    """Join of meets. Meet over an empty alternative is TOP (identity of
    meet); join over zero alternatives is BOTTOM — callers must not pass
    zero alternatives, since derived edges require a non-empty
    justification (graph.py enforces this at insertion).
    """
    result = BOTTOM
    for alt in alternatives:
        acc = TOP
        for s in alt:
            acc = meet(acc, s)
        result = join(result, acc)
    return result
