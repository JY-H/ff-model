"""Backtest 1: replay this league's completed 2026 weeks.

For every team and week, the decision is made at Sunday 11:45 ET: players whose
game kicked off earlier (Thursday, international) are locked in the slot they
actually played at their actual points; players who ended up inactive are
unavailable (inactive lists are public before each game's lock). Inputs are
ESPN's projections (the same numbers the league showed) plus nflverse context,
and the live marginals/correlations, which were fit on 2019-2025 only. The
opponent's lineup is what they actually started.

Compared per team-week: the model's lineup, the highest-projection lineup, and
the lineup the team actually started.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, time, timezone

import polars as pl

from ..config import DB_PATH, ET, LEAGUE_ID, SEASON
from ..research.factor_tests import factor_frame
from . import correlation as C
from .marginals import MODEL_PATH, Marginals, espn_context_quantiles
from .optimize import BENCH, IR, _slot_counts, decide

DECISION = time(11, 45)


def _decision_time(con, week: int) -> datetime:
    """Sunday 11:45 ET of that week (the date of its Sunday 1pm games)."""
    ks = [datetime.fromisoformat(r[0]).astimezone(ET) for r in
          con.execute("SELECT kickoff_utc FROM pro_games WHERE season=? AND week=?", (SEASON, week))]
    sunday = next(k.date() for k in sorted(ks) if k.weekday() == 6)
    return datetime.combine(sunday, DECISION, ET)


def _team_kickoffs(con) -> pl.DataFrame:
    from ..archive.context import espn_abbr
    abbr = espn_abbr(con)
    rows = []
    for r in con.execute("SELECT week, kickoff_utc, home_pro_team_id, away_pro_team_id FROM pro_games WHERE season=?", (SEASON,)):
        for t in (r[2], r[3]):
            rows.append((r[0], abbr.get(t), r[1]))
    return pl.DataFrame(rows, schema=["week", "team", "kickoff_utc"], orient="row").with_columns(
        pl.col("week").cast(pl.Int64), pl.col("kickoff_utc").str.to_datetime(time_zone="UTC"))


def final_weeks(con) -> set[int]:
    """Weeks whose final lineups and points are archived: a next-week Tuesday snapshot or a backfill."""
    return {r[0] for r in con.execute(
        "SELECT DISTINCT l.scoring_period FROM lineups l JOIN snapshots s ON s.id = l.snapshot_id"
        " WHERE l.league_id=? AND l.actual_points IS NOT NULL AND (s.week = l.scoring_period + 1 OR s.slot_key LIKE '%backfill')"
        " AND EXISTS (SELECT 1 FROM raw_responses r WHERE r.snapshot_id = s.id AND r.kind='prev_week' AND r.status_code=200)",
        (LEAGUE_ID,))}


def _final_rosters(con, week: int) -> pl.DataFrame:
    """Locked lineups and final points for every team (latest archive of that scoring period)."""
    sid = con.execute("SELECT MAX(snapshot_id) FROM lineups WHERE league_id=? AND scoring_period=? AND actual_points IS NOT NULL",
                      (LEAGUE_ID, week)).fetchone()[0]
    rows = con.execute("SELECT team_id, player_id, lineup_slot_id, actual_points FROM lineups WHERE league_id=? AND snapshot_id=?"
                       " AND scoring_period=?", (LEAGUE_ID, sid, week)).fetchall()
    mus = con.execute("SELECT home_team_id, away_team_id FROM matchups WHERE league_id=? AND snapshot_id=? AND matchup_period=?",
                      (LEAGUE_ID, sid, week)).fetchall()
    opp = {}
    for h, a in mus:
        opp[h], opp[a] = a, h
    return pl.DataFrame([dict(team_id=r[0], player_id=r[1], slot=r[2], actual=r[3] or 0.0, opp_team_id=opp.get(r[0]))
                         for r in rows])


def run(weeks=range(1, 10), n: int = 10000) -> pl.DataFrame:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    counts = _slot_counts(con)
    corr = C.load()
    marg = Marginals.load(MODEL_PATH)
    frame = factor_frame([SEASON], depth=50.0, with_u=False).filter(pl.col("week").is_in(list(weeks)))
    frame = C.assign_roles(frame)
    q_all = espn_context_quantiles(marg, frame)
    kick = _team_kickoffs(con)
    frame = frame.with_row_index("qi").join(kick, on=["week", "team"], how="left")
    out = []
    for week in weeks:
        dt = _decision_time(con, week)
        fw = frame.filter(pl.col("week") == week).select(
            pl.col("espn_id").alias("player_id"), "name", "pos", "proj", "team", "opp", "game", "role", "played",
            "kickoff_utc", "qi")
        ros = _final_rosters(con, week).join(fw, on="player_id", how="left").with_columns(
            pl.col("proj").fill_null(0.0),
            (pl.col("kickoff_utc") < pl.lit(dt.astimezone(timezone.utc))).fill_null(False).alias("locked"))
        ros = ros.with_columns((~pl.col("locked") & (pl.col("pos").is_null() | ~pl.col("played").fill_null(False)
                                                     | (pl.col("slot") == IR))).alias("unavailable"))
        q_of = {p: q_all[i] for p, i in zip(ros["player_id"], ros["qi"]) if i is not None}
        for team_id in sorted(ros["team_id"].unique()):
            me = ros.filter(pl.col("team_id") == team_id)
            opp_id = me["opp_team_id"][0]
            opp = ros.filter(pl.col("team_id") == opp_id)
            opp_start = opp.filter(~pl.col("slot").is_in([BENCH, IR]))
            # an opponent starter we can't simulate (no projection) contributes his actual points as fixed
            opp_start = opp_start.with_columns((pl.col("locked") | pl.col("qi").is_null()).alias("locked"))
            dec = decide(me, opp_start, q_of, counts, corr, n=n, seed=week * 100 + team_id)
            act_ids = set(me.filter(~pl.col("slot").is_in([BENCH, IR]) & ~pl.col("locked"))["player_id"])
            actual_of = dict(zip(me["player_id"], me["actual"]))
            opp_real = float(opp_start["actual"].sum())

            def realized(ids):
                return dec.fixed_me + sum(actual_of.get(p, 0.0) for p in ids)

            def pwin_of(ids):
                cols = [dec.col[p] for p in ids if p in dec.col]
                tot = dec.fixed_me + dec.D[:, cols].sum(axis=1)
                d = tot - dec.opp_total
                return float((d > 0).mean() + 0.5 * (d == 0).mean())

            ids = {k: {p for _, p in dec.lineups[i]} for k, i in (("model", dec.best), ("maxproj", dec.max_proj))}
            ids["actual"] = act_ids
            row = {"week": week, "team_id": team_id, "opp_team_id": opp_id, "opp_points": opp_real}
            for k, s in ids.items():
                pts = realized(s)
                row[f"{k}_points"] = pts
                row[f"{k}_win"] = float(pts > opp_real) + 0.5 * float(pts == opp_real)
                row[f"{k}_pwin"] = pwin_of(s)
            row["model_differs"] = ids["model"] != ids["maxproj"]
            row["n_changes"] = len(ids["model"] - ids["maxproj"])
            out.append(row)
    return pl.DataFrame(out)
