import numpy as np
import polars as pl

from ff_model.model.optimize import BENCH, SLOT_ID, _enumerate, _opponent_starters, decide, score_lineups

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


def test_every_game_started_means_nothing_to_simulate():
    # late in the week every starter on both sides is locked: the result is the locked points
    pos = ["QB", "RB", "RB", "WR", "WR", "TE", "WR", "K", "D/ST"]
    side = lambda base, pts: pl.DataFrame({  # noqa: E731
        "player_id": [base + i for i in range(9)], "pos": pos, "proj": [10.0] * 9, "slot": [0, 2, 2, 4, 4, 6, 23, 17, 16],
        "locked": [True] * 9, "unavailable": [False] * 9, "actual": [pts] * 9, "game": ["g"] * 9,
        "role": ["x"] * 9, "team": ["A"] * 9, "opp": ["B"] * 9})
    d = decide(side(0, 12.0), side(100, 10.0), {}, COUNTS, {}, n=100)
    assert d.fixed_me == 108.0 and float(d.opp_total.mean()) == 90.0 and d.pwin[d.best] == 1.0


def test_opponent_plays_their_best_lineup_not_the_one_they_set():
    # set: QB 1 (Out), RB 2 3, WR 4 5, TE 6, FLEX 7, K 8, D/ST 9; bench: QB 10, RB 11 (better than 3), TE 12 (played Thu, benched)
    S = SLOT_ID
    opp = pl.DataFrame({
        "player_id": list(range(1, 13)),
        "pos": ["QB", "RB", "RB", "WR", "WR", "TE", "WR", "K", "D/ST", "QB", "RB", "TE"],
        "proj": [20.0, 15, 6, 14, 12, 8, 7, 8, 7, 16, 9, 30],
        "slot": [S["QB"], S["RB"], S["RB"], S["WR"], S["WR"], S["TE"], S["FLEX"], S["K"], S["D/ST"], BENCH, BENCH, BENCH],
        "locked": [False] * 11 + [True], "unavailable": [True] + [False] * 11})
    best = dict(zip(*_opponent_starters(opp, COUNTS).select("player_id", "slot")))
    assert set(best) == {10, 2, 11, 4, 5, 6, 7, 8, 9}  # QB 10 for the Out QB; bench RB 11 replaces RB 3
    assert best[10] == S["QB"] and best[11] == S["RB"] and best[7] == S["FLEX"] and 12 not in best  # locked bench TE stays out
    as_set = dict(zip(*_opponent_starters(opp, COUNTS, as_set=True).select("player_id", "slot")))
    assert set(as_set) == {10, 2, 3, 4, 5, 6, 7, 8, 9} and as_set[10] == S["QB"]  # only the Out QB is replaced
