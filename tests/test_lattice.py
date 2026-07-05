from hypothesis import given
from hypothesis import strategies as st

from bte import ALL_SIGMA, BOTTOM, TOP, S, Sigma, evaluate, join, leq, meet

sigmas = st.sampled_from(ALL_SIGMA)
alt_lists = st.lists(
    st.lists(sigmas, min_size=1, max_size=4), min_size=1, max_size=4
)


@given(sigmas, sigmas)
def test_meet_join_are_bounds(a, b):
    assert leq(meet(a, b), a) and leq(meet(a, b), b)
    assert leq(a, join(a, b)) and leq(b, join(a, b))


@given(sigmas)
def test_bounds(a):
    assert leq(BOTTOM, a) and leq(a, TOP)


def _case_split(alts):
    """The three-case definition from formalism.md section 3, per axis."""
    out = []
    for axis in range(2):
        vals = [[s[axis] for s in alt] for alt in alts]
        if any(all(v == S.TOP for v in alt) for alt in vals):
            out.append(S.TOP)
        elif all(any(v == S.BOT for v in alt) for alt in vals):
            out.append(S.BOT)
        else:
            out.append(S.U)
    return Sigma(*out)


@given(alt_lists)
def test_evaluate_matches_case_split(alts):
    assert evaluate(alts) == _case_split(alts)


@given(alt_lists, st.data())
def test_evaluate_is_monotone(alts, data):
    """Raising one input never lowers the output (the property every
    theorem in formalism.md rests on)."""
    i = data.draw(st.integers(0, len(alts) - 1))
    j = data.draw(st.integers(0, len(alts[i]) - 1))
    raised = data.draw(st.sampled_from(
        [s for s in ALL_SIGMA if leq(alts[i][j], s)]))
    bumped = [list(alt) for alt in alts]
    bumped[i][j] = raised
    assert leq(evaluate(alts), evaluate(bumped))
