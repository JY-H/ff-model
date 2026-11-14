import numpy as np
import polars as pl

from ff_model.model.optimize import _enumerate, score_lineups

COUNTS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "D/ST": 1}


def test_enumerated_lineups_are_legal_and_unique():
    free = pl.DataFrame({"player_id": list(range(12)),
                         "pos": ["QB", "QB", "RB", "RB", "RB", "WR", "WR", "WR", "WR", "TE", "K", "D/ST"]})
    lus = _enumerate(free, COUNTS)
    sets = [frozenset(p for _, p in lu) for lu in lus]
    assert len(sets) == len(set(sets))
    pos = dict(zip(free["player_id"], free["pos"]))
    for lu in lus:
        slots = [s for s, _ in lu]
        assert sorted(slots) == sorted(sum(([k] * v for k, v in COUNTS.items()), []))
        for s, p in lu:
            assert pos[p] in ({"RB", "WR", "TE"} if s == "FLEX" else {s})
    # distinct player sets: 5 of the 7 RB/WR with >= 2 of each (C(7,5) - 3 = 18), x 2 QBs
    assert len(lus) == 36


def test_favorite_prefers_steady_underdog_prefers_volatile():
    rng = np.random.default_rng(0)
    n = 200_000
    steady = 12 + 2 * rng.standard_normal(n)     # same mean, low variance
    volatile = 12 + 9 * rng.standard_normal(n)   # same mean, high variance
    D = np.column_stack([steady, volatile])
    L = np.array([[1, 0], [0, 1]])
    opp = 100 + 5 * rng.standard_normal(n)
    _, _, p_fav = score_lineups(D, L, fixed_me=96.0, opp_total=opp)   # favored by ~8
    _, _, p_dog = score_lineups(D, L, fixed_me=80.0, opp_total=opp)   # trailing by ~8
    assert p_fav[0] > p_fav[1]   # favorite: steady player
    assert p_dog[1] > p_dog[0]   # underdog: volatile player


def test_short_roster_gets_an_empty_slot_not_no_lineup():
    free = pl.DataFrame({"player_id": [1, 2, 3, 4, 5, 6, 7, 8], "pos": ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "D/ST"]})
    lus = _enumerate(free, COUNTS)   # no kicker available
    assert lus and all(any(s == "K" and isinstance(p, str) for s, p in lu) for lu in lus)
