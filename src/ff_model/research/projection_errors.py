"""Join ESPN's historical weekly projections to actual results, both in league scoring.

One row per projected player-week: proj (ESPN, rescored to this league), actual
(nflverse, league-scored; 0 if the player didn't appear in the stats).
"""
from __future__ import annotations

import polars as pl

from ..db import connect, load_raw
from . import espn_history, scoring
from .nflverse import load

POS = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "D/ST"}
# ESPN pro-team abbreviations that differ from nflverse
ABBR_FIX = {"WSH": "WAS", "LAR": "LA"}


def _espn_team_abbr(con) -> dict[int, str]:
    row = con.execute("SELECT * FROM raw_responses WHERE kind='pro_schedule' AND status_code=200"
                      " ORDER BY id DESC LIMIT 1").fetchone()
    teams = load_raw(row)["settings"]["proTeams"]
    return {t["id"]: ABBR_FIX.get(t["abbrev"].upper(), t["abbrev"].upper()) for t in teams}


def build(seasons: list[int]) -> pl.DataFrame:
    con = connect()
    rules = scoring.league_rules(con)
    proj = scoring.score_wide(espn_history.parse(seasons, write=False), rules)
    proj = proj.select("season", "week", "espn_id", "name", "position_id", "pro_team_id",
                       pl.col("fpts").alias("proj"))

    ids = (load("ff_playerids").select(pl.col("espn_id").cast(pl.Int64, strict=False), "gsis_id")
           .drop_nulls().unique("espn_id"))
    ps = scoring.score_players(load("player_stats", seasons).filter(pl.col("season_type") == "REG"), rules)
    players = (proj.filter(pl.col("position_id") != 16)
               .join(ids, on="espn_id", how="left")
               .join(ps.select("season", "week", pl.col("player_id").alias("gsis_id"),
                               pl.col("fpts").alias("actual")),
                     on=["season", "week", "gsis_id"], how="left"))

    abbr = _espn_team_abbr(con)
    d = scoring.score_dst(scoring.dst_inputs(load("team_stats", seasons), load("schedules", seasons)), rules)
    d = d.with_columns(pl.when(pl.col("team") == "OAK").then(pl.lit("LV")).otherwise(pl.col("team")).alias("team"))
    team_ids = pl.DataFrame({"espn_id": [-16000 - k for k in abbr], "team": list(abbr.values())})
    dst = (proj.filter(pl.col("position_id") == 16).join(team_ids, on="espn_id", how="left")
           .join(d.select("season", "week", "team", pl.col("fpts").alias("actual")),
                 on=["season", "week", "team"], how="left"))

    out = pl.concat([players.drop("gsis_id"), dst.drop("team")], how="diagonal_relaxed")
    return out.with_columns(
        pl.col("position_id").replace_strict(POS, default="other").alias("pos"),
        pl.col("actual").is_not_null().alias("played"),
        pl.col("actual").fill_null(0.0),
    ).with_columns((pl.col("actual") - pl.col("proj")).alias("err"))


# Weekly "fantasy-relevant" pool for a 16-team league: ~1.5x starters by ESPN projection rank
RELEVANT_TOP_N = {"QB": 24, "RB": 56, "WR": 64, "TE": 24, "K": 24, "D/ST": 24}


def relevant(df: pl.DataFrame) -> pl.DataFrame:
    ranked = df.with_columns(pl.col("proj").rank("ordinal", descending=True)
                             .over(["season", "week", "pos"]).alias("pos_rank"))
    caps = pl.col("pos").replace_strict(RELEVANT_TOP_N, default=0)
    return ranked.filter((pl.col("pos_rank") <= caps) & (pl.col("proj") > 0))


def summary(df: pl.DataFrame) -> pl.DataFrame:
    return df.group_by("pos").agg(
        n=pl.len(),
        mean_proj=pl.col("proj").mean(),
        bias=pl.col("err").mean(),                 # actual - proj; + = ESPN too low
        mae=pl.col("err").abs().mean(),
        rmse=(pl.col("err") ** 2).mean().sqrt(),
        corr=pl.corr("proj", "actual"),
        pct_within_5=(pl.col("err").abs() <= 5).mean() * 100,
        p10_err=pl.col("err").quantile(0.10),
        p90_err=pl.col("err").quantile(0.90),
        skew=pl.col("err").skew(),
        zero_or_dnp=((pl.col("actual") <= 0) | ~pl.col("played")).mean() * 100,
    ).sort(pl.col("pos").replace_strict({p: i for i, p in enumerate(RELEVANT_TOP_N)}, default=99))


def calibration(df: pl.DataFrame, edges=(0, 5, 10, 15, 20, 25, 99)) -> pl.DataFrame:
    labels = [f"{a}-{b}" if b < 99 else f"{a}+" for a, b in zip(edges[:-1], edges[1:])]
    return (df.with_columns(pl.col("proj").cut(list(edges[1:-1]), labels=labels, left_closed=True).alias("bucket"))
              .group_by("bucket").agg(n=pl.len(), mean_proj=pl.col("proj").mean(),
                                      mean_actual=pl.col("actual").mean(), sd_err=pl.col("err").std(),
                                      p10_actual=pl.col("actual").quantile(0.1),
                                      p90_actual=pl.col("actual").quantile(0.9))
              .sort("mean_proj"))
