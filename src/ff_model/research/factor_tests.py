"""Screen candidate factors: do they predict ESPN's miss, or change the range?

Every factor uses only information available before the game:
season-to-date stats from prior weeks, closing lines, and game-time weather
(a stand-in for the forecast we'd have at lock).

For each (factor, position, group):
  shift     = mean(actual - proj) in the group minus the position's overall mean
  ci95      = 1.96 x standard error of the group mean
  seasons   = how many seasons the shift has the same sign as overall
  bust/boom = share landing below p10 / above p90 of the player's predicted range
"""
from __future__ import annotations

import numpy as np
import polars as pl

from . import scoring
from .nflverse import load
from .projection_errors import _espn_team_abbr, build, relevant
from ..db import connect
from ..model.marginals import MODEL_PATH, Marginals, cdf

SEASONS = list(range(2019, 2026))
MIN_PRIOR_GAMES = 3


def _team_games() -> pl.DataFrame:
    """One row per team-game: opponent, home/away, weather, lines, implied total."""
    s = load("schedules", SEASONS).filter(pl.col("game_type") == "REG")
    base = ["season", "week", "roof", "temp", "wind", "spread_line", "total_line"]
    home = s.select(*base, pl.col("home_team").alias("team"), pl.col("away_team").alias("opp"),
                    pl.lit(1).alias("is_home"))
    away = s.select(*base, pl.col("away_team").alias("team"), pl.col("home_team").alias("opp"),
                    pl.lit(0).alias("is_home"))
    g = pl.concat([home, away])
    # nflverse spread_line > 0 means the home team is favored by that many points
    margin = pl.when(pl.col("is_home") == 1).then(pl.col("spread_line")).otherwise(-pl.col("spread_line"))
    return g.with_columns(margin.alias("exp_margin"),
                          ((pl.col("total_line") + margin) / 2).alias("implied_total"))


def _trailing(df: pl.DataFrame, key: str, cols: list[str]) -> pl.DataFrame:
    """Season-to-date per-game means over *prior* weeks only, per key (team)."""
    df = df.with_columns(pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64)).sort("season", key, "week")
    out = df.select("season", "week", key)
    for c in cols:
        prior_sum = pl.col(c).cum_sum().over("season", key) - pl.col(c)
        out = out.with_columns(df.select(prior_sum.alias(f"_{c}"))[f"_{c}"])
    n_prior = df.select((pl.col("week").cum_count().over("season", key) - 1).alias("n"))["n"]
    out = out.with_columns(n_prior.alias("n_prior"))
    return out.with_columns([(pl.col(f"_{c}") / pl.col("n_prior")).alias(f"{c}_pg") for c in cols]).drop(
        [f"_{c}" for c in cols])


def factor_frame() -> pl.DataFrame:
    con = connect()
    rules = scoring.league_rules(con)
    df = relevant(build(SEASONS))

    # Player's NFL team that week. ESPN's historical entries don't carry it (proTeamId is 0, and
    # the player-level team is *current*), so use nflverse weekly rosters; D/ST ids encode the team.
    ids = (load("ff_playerids").select(pl.col("espn_id").cast(pl.Int64, strict=False), "gsis_id")
           .drop_nulls().unique("espn_id"))
    rost = (load("rosters_weekly", SEASONS)
            .select(pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64), "gsis_id",
                    pl.col("team").alias("roster_team"))
            .drop_nulls("gsis_id").unique(["season", "week", "gsis_id"]))
    abbr = _espn_team_abbr(con)
    dst_team = (pl.col("espn_id") * -1 - 16000).replace_strict(abbr, default=None)
    dst_team = pl.when((pl.col("season") == 2019) & (dst_team == "LV")).then(pl.lit("OAK")).otherwise(dst_team)
    df = (df.join(ids, on="espn_id", how="left")
            .join(rost, on=["season", "week", "gsis_id"], how="left")
            .with_columns(pl.when(pl.col("pos") == "D/ST").then(dst_team)
                          .otherwise(pl.col("roster_team")).alias("team")))
    g = _team_games().with_columns(pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64))
    df = df.join(g, on=["season", "week", "team"], how="left")

    # --- offense pace and red-zone stalls (team, prior weeks) ---
    ts = load("team_stats", SEASONS).filter(pl.col("season_type") == "REG").with_columns(
        (pl.col("attempts") + pl.col("carries") + pl.col("sacks_suffered")).alias("plays"),
        (pl.col("passing_tds") + pl.col("rushing_tds")).alias("off_tds"),
        pl.col("fg_att").cast(pl.Float64),
    )
    off = _trailing(ts, "team", ["plays", "off_tds", "fg_att", "passing_yards", "rushing_yards"])
    off = off.with_columns((pl.col("fg_att_pg") / (pl.col("fg_att_pg") + pl.col("off_tds_pg"))).alias("stall_rate"))
    df = df.join(off.select("season", "week", "team", "n_prior", "plays_pg", "stall_rate"),
                 on=["season", "week", "team"], how="left")

    # --- opponent defense: yards allowed per game (prior weeks) ---
    allowed = ts.select("season", "week", pl.col("opponent_team").alias("defense"),
                        pl.col("passing_yards").alias("pass_yds_allowed"),
                        pl.col("rushing_yards").alias("rush_yds_allowed"))
    dfn = _trailing(allowed, "defense", ["pass_yds_allowed", "rush_yds_allowed"])
    df = df.join(dfn.select("season", "week", pl.col("defense").alias("opp"),
                            pl.col("n_prior").alias("opp_n_prior"), "pass_yds_allowed_pg", "rush_yds_allowed_pg"),
                 on=["season", "week", "opp"], how="left")

    # --- fantasy points allowed to the position (prior weeks) ---
    ps = scoring.score_players(load("player_stats", SEASONS).filter(pl.col("season_type") == "REG"), rules)
    fpa = (ps.filter(pl.col("position").is_in(["QB", "RB", "WR", "TE", "K"]))
             .group_by("season", "week", "opponent_team", "position").agg(pl.col("fpts").sum().alias("fpa")))
    fpa_t = pl.concat([
        _trailing(fpa.filter(pl.col("position") == p).rename({"opponent_team": "defense"}), "defense", ["fpa"])
        .with_columns(pl.lit(p).alias("pos")) for p in ["QB", "RB", "WR", "TE", "K"]])
    df = df.join(fpa_t.select("season", "week", pl.col("defense").alias("opp"), "pos", "fpa_pg"),
                 on=["season", "week", "opp", "pos"], how="left")

    # --- injury designation that week (players who were listed) ---
    inj = (load("injuries", SEASONS).filter(pl.col("game_type") == "REG")
           .select(pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64), "gsis_id", "report_status")
           .unique(["season", "week", "gsis_id"]))
    df = (df.join(inj, on=["season", "week", "gsis_id"], how="left")
            .with_columns(pl.col("report_status").fill_null("none")))

    # --- where the actual landed in the player's predicted range ---
    m = Marginals.load(MODEL_PATH)
    df = df.with_columns(pl.Series("u", cdf(m.quantiles(df), df["actual"].to_numpy())))
    return df


def _groups(df: pl.DataFrame) -> list[tuple[str, pl.Expr, list[str]]]:
    """(factor name, group expression, positions to test)."""
    outdoor = pl.col("roof").is_in(["outdoors", "open"])
    wind = (pl.when(~outdoor).then(pl.lit("dome/closed"))
              .when(pl.col("wind") >= 15).then(pl.lit("wind 15+"))
              .when(pl.col("wind") >= 10).then(pl.lit("wind 10-14"))
              .otherwise(pl.lit("wind <10")))
    cold = (pl.when(~outdoor).then(pl.lit("dome/closed"))
              .when(pl.col("temp") <= 32).then(pl.lit("freezing (<=32F)"))
              .when(pl.col("temp") <= 45).then(pl.lit("cold (33-45F)"))
              .otherwise(pl.lit("mild (>45F)")))
    enough = pl.col("n_prior") >= MIN_PRIOR_GAMES
    opp_enough = pl.col("opp_n_prior") >= MIN_PRIOR_GAMES

    def terc(col, cond, labels=("low", "mid", "high")):
        # terciles computed within position
        q1 = pl.col(col).quantile(1 / 3).over("pos")
        q2 = pl.col(col).quantile(2 / 3).over("pos")
        return (pl.when(~cond | pl.col(col).is_null()).then(pl.lit(None))
                  .when(pl.col(col) <= q1).then(pl.lit(labels[0]))
                  .when(pl.col(col) <= q2).then(pl.lit(labels[1])).otherwise(pl.lit(labels[2])))

    skill = ["QB", "RB", "WR", "TE"]
    return [
        ("wind", wind, ["QB", "WR", "TE", "RB", "K"]),
        ("temperature", cold, ["QB", "WR", "TE", "RB", "K"]),
        ("opp pass yds allowed", terc("pass_yds_allowed_pg", opp_enough, ("stingy", "avg", "generous")), ["QB", "WR", "TE"]),
        ("opp rush yds allowed", terc("rush_yds_allowed_pg", opp_enough, ("stingy", "avg", "generous")), skill),
        ("opp pts allowed to pos", terc("fpa_pg", opp_enough, ("stingy", "avg", "generous")), skill + ["K"]),
        ("team pace (plays/g)", terc("plays_pg", enough, ("slow", "avg", "fast")), skill + ["K"]),
        ("expected margin", pl.when(pl.col("exp_margin") >= 7).then(pl.lit("fav 7+"))
                              .when(pl.col("exp_margin") <= -7).then(pl.lit("dog 7+"))
                              .when(pl.col("exp_margin").is_not_null()).then(pl.lit("within 7")), skill + ["K"]),
        ("implied team total", terc("implied_total", pl.lit(True)), ["K"]),
        ("team stall rate (FG/(FG+TD))", terc("stall_rate", enough), ["K"]),
        ("injury designation", pl.when(pl.col("report_status").is_in(["Questionable", "Doubtful"]))
                                 .then(pl.lit("Q/D")).when(pl.col("report_status") == "none")
                                 .then(pl.lit("not listed")), skill),
    ]


def screen(df: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for name, grp, positions in _groups(df):
        d = df.with_columns(grp.alias("g")).filter(pl.col("pos").is_in(positions) & pl.col("g").is_not_null())
        for pos in positions:
            dp = d.filter(pl.col("pos") == pos)
            if dp.height == 0:
                continue
            base = dp["err"].mean()
            base_by_season = dp.group_by("season").agg(pl.col("err").mean().alias("b"))
            for g in sorted(dp["g"].unique().to_list()):
                x = dp.filter(pl.col("g") == g)
                e = x["err"].to_numpy()
                shift = float(e.mean() - base)
                se = float(e.std(ddof=1) / np.sqrt(len(e))) if len(e) > 1 else np.nan
                by = (x.group_by("season").agg(pl.col("err").mean().alias("m"), pl.len().alias("k"))
                        .join(base_by_season, on="season").filter(pl.col("k") >= 10))
                agree = int(((by["m"] - by["b"]) * np.sign(shift) > 0).sum())
                u = x["u"].to_numpy()
                rows.append({"factor": name, "pos": pos, "group": g, "n": len(e),
                             "shift": round(shift, 2), "ci95": round(1.96 * se, 2),
                             "seasons": f"{agree}/{by.height}",
                             "bust%": round(100 * float((u < 0.1).mean()), 1),
                             "boom%": round(100 * float((u > 0.9).mean()), 1)})
    return pl.DataFrame(rows)
