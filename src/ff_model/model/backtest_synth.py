"""Backtest 2: synthetic leagues built from real 2022-2025 player-weeks, walk-forward.

For each evaluated season, marginals and correlations are fit only on earlier
seasons. Each week, the fantasy-relevant pool is dealt at random to 16 teams of
16 (2 QB, 5 RB, 5 WR, 2 TE, K, D/ST), paired into 8 matchups, `leagues` times.
Every team picks (a) the model's best-P(win) lineup and (b) the highest-projection
lineup against an opponent starting its highest-projection lineup; both are
scored on what the players actually scored. Inactive players are excluded
(known at lock); every game is treated as not yet started.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import polars as pl
from scipy.stats import norm

from ..config import DATA_DIR
from . import correlation as C
from .marginals import Marginals, cdf
from .optimize import BENCH, decide

log = logging.getLogger("ff_model.backtest_synth")
COUNTS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "D/ST": 1}
POOL = {"QB": 32, "RB": 80, "WR": 80, "TE": 32, "K": 16, "D/ST": 16}   # dealt to 16 teams
PER_TEAM = {p: n // 16 for p, n in POOL.items()}                       # 2/5/5/2/1/1 = 16
OUT_DIR = DATA_DIR / "backtest_synth"


def _max_proj_lineup(roster: pl.DataFrame) -> list[int]:
    picked = []
    for pos, k in (("QB", 1), ("RB", 2), ("WR", 2), ("TE", 1), ("K", 1), ("D/ST", 1)):
        picked += roster.filter(pl.col("pos") == pos).sort("proj", descending=True)["player_id"].head(k).to_list()
    rest = roster.filter(pl.col("pos").is_in(["RB", "WR", "TE"]) & ~pl.col("player_id").is_in(picked))
    return picked + rest.sort("proj", descending=True)["player_id"].head(1).to_list()


def run_season(frame: pl.DataFrame, season: int, leagues: int = 10, n: int = 4000, seed: int = 0) -> pl.DataFrame:
    train = frame.filter(pl.col("season") < season)
    t0 = time.time()
    marg = Marginals(use_context=True).fit(train)
    u = cdf(marg.quantiles(train), train["actual"].to_numpy(), np.random.default_rng(0))
    corr = C.estimate(C.assign_roles(train.with_columns(pl.Series("z", norm.ppf(np.clip(u, 1e-4, 1 - 1e-4))))))
    log.info("season %d: fit on %d rows in %.0fs", season, train.height, time.time() - t0)

    test = C.assign_roles(frame.filter(pl.col("season") == season)).with_columns(
        pl.col("espn_id").alias("player_id"), pl.lit(BENCH).alias("slot"), pl.lit(False).alias("locked"),
        pl.lit(False).alias("unavailable"))
    q = marg.quantiles(test)
    q_of = dict(zip(test["player_id"].to_list(), q))
    rng = np.random.default_rng(seed + season)
    rows = []
    for week in sorted(test["week"].unique()):
        wk = test.filter(pl.col("week") == week)
        pools = {p: wk.filter(pl.col("pos") == p).sort("proj", descending=True).head(POOL[p]) for p in POOL}
        if any(pools[p].height < POOL[p] for p in POOL):
            continue
        for lg in range(leagues):
            teams = [[] for _ in range(16)]
            for p, d in pools.items():
                ids = d["player_id"].to_numpy().copy()
                rng.shuffle(ids)
                for t in range(16):
                    teams[t] += ids[t * PER_TEAM[p]:(t + 1) * PER_TEAM[p]].tolist()
            order = rng.permutation(16)
            for a, b in zip(order[::2], order[1::2]):
                for me_t, opp_t in ((a, b), (b, a)):
                    me = wk.filter(pl.col("player_id").is_in(teams[me_t]))
                    opp = wk.filter(pl.col("player_id").is_in(teams[opp_t]))
                    opp_start = opp.filter(pl.col("player_id").is_in(_max_proj_lineup(opp)))
                    dec = decide(me, opp_start, q_of, COUNTS, corr, n=n, seed=int(rng.integers(1 << 31)))
                    act = dict(zip(me["player_id"], me["actual"]))
                    opp_pts = float(opp_start["actual"].sum())
                    m_ids = {p for _, p in dec.lineups[dec.best]}
                    x_ids = {p for _, p in dec.lineups[dec.max_proj]}
                    m_pts = sum(act.get(p, 0.0) for p in m_ids)
                    x_pts = sum(act.get(p, 0.0) for p in x_ids)
                    rows.append({"season": season, "week": week, "league": lg, "team": int(me_t),
                                 "differs": m_ids != x_ids, "n_changes": len(m_ids - x_ids),
                                 "model_pwin": float(dec.pwin[dec.best]), "maxproj_pwin": float(dec.pwin[dec.max_proj]),
                                 "model_points": m_pts, "maxproj_points": x_pts, "opp_points": opp_pts,
                                 "model_win": float(m_pts > opp_pts) + 0.5 * float(m_pts == opp_pts),
                                 "maxproj_win": float(x_pts > opp_pts) + 0.5 * float(x_pts == opp_pts)})
        log.info("season %d week %d done (%d decisions so far, %.0fs)", season, week, len(rows), time.time() - t0)
    out = pl.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT_DIR / f"{season}.parquet")
    return out


def run(seasons=(2022, 2023, 2024, 2025), leagues: int = 10, n: int = 4000) -> None:
    frame = pl.read_parquet(DATA_DIR / "marginal_frame.parquet")  # 2019-2025, active players, with context
    for s in seasons:
        if (OUT_DIR / f"{s}.parquet").exists():
            log.info("season %d already done", s)
            continue
        run_season(frame, s, leagues, n)


def summary() -> pl.DataFrame:
    df = pl.concat([pl.read_parquet(f) for f in sorted(OUT_DIR.glob("*.parquet"))])
    return df
