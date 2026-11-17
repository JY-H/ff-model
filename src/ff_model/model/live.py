"""Assemble live model inputs for one archived snapshot.

Everything comes from what was archived at (or before) that snapshot, so the
same code serves live decisions and leak-free backtests:
  projection + stat line + ESPN injury designation  <- player_snapshots
  opponent, kickoff                                  <- pro_games
  stadium, roof, kickoff-window forecast             <- game_context
  expected margin                                    <- latest odds_lines at or before the snapshot
  practice participation                             <- injury_reports (nflverse)
  games back after an absence                        <- nflverse weekly rosters (prior weeks)
"""
from __future__ import annotations

import json
import sqlite3

import polars as pl

from ..archive.context import espn_abbr
from ..config import DB_PATH, LEAGUE_ID, SEASON
from ..research.factor_tests import _add_line_injuries, _add_return_from_absence
from ..research.line_injuries import starters_out
from ..research.nflverse import load
from ..research.projection_errors import POS
from .marginals import MODEL_PATH, TAUS, Marginals, espn_context_quantiles

# Odds API full names -> nflverse abbreviations
CONTEXT_TYPES = {"roof": pl.String, "temp": pl.Float64, "wind": pl.Float64, "exp_margin": pl.Float64,
                 "practice_status": pl.String, "injury": pl.String, "games_back": pl.Int64,
                 "ol_out": pl.Int64, "opp_ol_out": pl.Int64, "opp_db_out": pl.Int64}

TEAM_NAMES = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}
ESPN_STATUS = {"QUESTIONABLE": "Questionable", "DOUBTFUL": "Doubtful", "OUT": "Out"}
LINE = {"3": "p_pass_yds", "4": "p_pass_td", "20": "p_int", "24": "p_rush_yds", "25": "p_rush_td",
        "42": "p_rec_yds", "43": "p_rec_td", "53": "p_rec"}


def _q(con, sql, args=()) -> pl.DataFrame:
    rows = con.execute(sql, args).fetchall()
    cols = [d[0] for d in con.execute(sql, args).description] if rows else []
    return pl.DataFrame([dict(zip(cols, r)) for r in rows]) if rows else pl.DataFrame()


def _latest_with(con, table: str, sid: int, week: int) -> int | None:
    r = con.execute(f"SELECT MAX(t.snapshot_id) FROM {table} t JOIN snapshots s ON s.id=t.snapshot_id"
                    f" WHERE t.snapshot_id <= ? AND s.week=?", (sid, week)).fetchone()
    return r[0]


def build_inputs(snapshot_id: int | None = None) -> pl.DataFrame:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    if snapshot_id is None:
        snapshot_id = con.execute("SELECT MAX(id) FROM snapshots WHERE status IN ('ok','partial')"
                                  " AND (slot_key IS NULL OR slot_key NOT LIKE '%backfill')").fetchone()[0]
    week = con.execute("SELECT week FROM snapshots WHERE id=?", (snapshot_id,)).fetchone()[0]
    abbr = espn_abbr(con)

    ps = _q(con, "SELECT player_id, full_name AS name, position_id, pro_team_id, injury_status, on_team_id,"
                 " proj_points AS proj, proj_stats_json FROM player_snapshots WHERE league_id=? AND snapshot_id=?"
                 " AND (proj_points > 0 OR on_team_id > 0)",  # rostered players always (locked/late swap)
            (LEAGUE_ID, snapshot_id))
    stats = [json.loads(s or "{}") for s in ps["proj_stats_json"]]
    df = ps.with_columns(
        *[pl.Series(v, [float(st.get(k, 0.0)) for st in stats]) for k, v in LINE.items()],
        pl.col("position_id").replace_strict(POS, default="other").alias("pos"),
        pl.col("pro_team_id").replace_strict(abbr, default=None).alias("team"),
        pl.col("injury_status").replace_strict(ESPN_STATUS, default="none").alias("report_status"),
        pl.lit(SEASON, dtype=pl.Int64).alias("season"), pl.lit(week, dtype=pl.Int64).alias("week"),
    ).drop("proj_stats_json")

    # opponent + kickoff
    g = _q(con, "SELECT game_id, kickoff_utc, home_pro_team_id AS h, away_pro_team_id AS a FROM pro_games"
                " WHERE season=? AND week=?", (SEASON, week))
    g = g.with_columns(pl.col("h").replace_strict(abbr, default=None).alias("home"),
                       pl.col("a").replace_strict(abbr, default=None).alias("away"))
    sides = pl.concat([g.select("game_id", "kickoff_utc", pl.col("home").alias("team"), pl.col("away").alias("opp")),
                       g.select("game_id", "kickoff_utc", pl.col("away").alias("team"), pl.col("home").alias("opp"))])
    df = df.join(sides, on="team", how="left")

    # weather / roof
    csid = _latest_with(con, "game_context", snapshot_id, week)
    if csid:
        gc = _q(con, "SELECT game_id, stadium, roof, temp_f AS temp, wind_mph AS wind, gust_mph, precip_prob"
                     " FROM game_context WHERE snapshot_id=?", (csid,))
        df = df.join(gc, on="game_id", how="left")

    # expected margin: median spread across books, latest odds at or before this snapshot
    osid = con.execute("SELECT MAX(snapshot_id) FROM odds_lines WHERE snapshot_id <= ?", (snapshot_id,)).fetchone()[0]
    if osid:
        od = _q(con, "SELECT outcome, point, commence_time FROM odds_lines WHERE snapshot_id=? AND market='spreads'",
                (osid,))
        od = (od.with_columns(pl.col("outcome").replace_strict(TEAM_NAMES, default=None).alias("team"),
                              pl.col("commence_time").str.to_datetime(time_zone="UTC").alias("ct"))
                .group_by("team", "ct").agg((-pl.col("point").median()).alias("exp_margin")))
        # keep the line for this week's game only (the odds feed lists future weeks too), matched per
        # team first, then joined without dropping anyone (byes, games already played)
        kick = (df.select("team", "kickoff_utc").drop_nulls().unique("team")
                  .with_columns(pl.col("kickoff_utc").str.to_datetime(time_zone="UTC").alias("_k")))
        line = (kick.join(od, on="team", how="inner")
                    .filter((pl.col("ct") - pl.col("_k")).abs() < pl.duration(hours=36))
                    .group_by("team").agg(pl.col("exp_margin").first()))
        df = df.join(line, on="team", how="left")

    # practice participation (nflverse), by gsis id
    ids = (load("ff_playerids").select(pl.col("espn_id").cast(pl.Int64, strict=False).alias("player_id"), "gsis_id")
           .drop_nulls().unique("player_id"))
    df = df.join(ids, on="player_id", how="left")
    isid = _latest_with(con, "injury_reports", snapshot_id, week)
    if isid:
        inj = _q(con, "SELECT gsis_id, practice_status, injury FROM injury_reports WHERE snapshot_id=?", (isid,))
        df = df.join(inj.unique("gsis_id"), on="gsis_id", how="left")

    df = _add_return_from_absence(df, [SEASON])

    # OL / DB starters out: this week from the archived injury report only
    if isid:
        rep = _q(con, "SELECT season, week, gsis_id, report_status FROM injury_reports WHERE snapshot_id=?", (isid,))
        rep = rep.with_columns(pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64))
        line = starters_out([SEASON], extra_injuries=rep, asof_week=week).filter(pl.col("week") == week)
        df = _add_line_injuries(df, [SEASON], line=line)

    # fixed types: early in the week (no injury report yet) or late (no outdoor game left) a column
    # can be empty for every player, and an all-null column can't take the neutral values later
    for c, t in CONTEXT_TYPES.items():
        df = df.with_columns((pl.col(c) if c in df.columns else pl.lit(None)).cast(t).alias(c))
    return df.with_columns(pl.lit(snapshot_id).alias("snapshot_id"))


def with_quantiles(df: pl.DataFrame, show=(0.1, 0.5, 0.9)) -> pl.DataFrame:
    q = espn_context_quantiles(Marginals.load(MODEL_PATH), df)
    taus = list(TAUS)
    return df.with_columns([pl.Series(f"p{int(t*100)}", q[:, taus.index(t)]) for t in show])
