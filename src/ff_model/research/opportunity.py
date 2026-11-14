"""Opportunity vs. result, from play-by-play: was a player's last week luck or role?

Per player-game (nflverse ff_opportunity + play-by-play, this league's scoring):
  xfp   expected fantasy points from the player's targets/carries/attempts, given where they happened
  fp    the same scoring applied to what actually happened (fumbles excluded from both)
  fpoe  fp - xfp: points over expected, i.e. efficiency/luck
  shares of team targets, air yards, and plays inside the 10

Features for a game use the player's *earlier* games that season only (no lookahead):
  *_l1 = the previous game, *_l3 = mean of up to the 3 previous games (needs 2+).
"""
from __future__ import annotations

import polars as pl

from .nflverse import load

# this league's rules on nflverse's expected/actual components, by part of the game
# (stat ids 53, 42, 43, 24, 25, 3, 4, 20 and the 2-pt conversions)
PARTS = {
    "rec": {"receptions": 1.0, "rec_yards_gained": 0.1, "rec_touchdown": 6.0, "rec_two_point_conv": 2.0},
    "rush": {"rush_yards_gained": 0.1, "rush_touchdown": 6.0, "rush_two_point_conv": 2.0},
    "pass": {"pass_yards_gained": 0.04, "pass_touchdown": 4.0, "pass_interception": -2.0, "pass_two_point_conv": 2.0},
}
_PTS = {k: v for part in PARTS.values() for k, v in part.items()}
# raw counts kept for the post-week review (actual and expected)
DETAIL = ["receptions", "rec_yards_gained", "rec_touchdown", "rush_yards_gained", "rush_touchdown",
          "pass_completions", "pass_yards_gained", "pass_touchdown", "pass_interception"]


def _pts(suffix: str, terms: dict | None = None) -> pl.Expr:
    return pl.sum_horizontal([pl.col(f"{k}{suffix}").fill_null(0) * v for k, v in (terms or _PTS).items()])


def player_games(seasons: list[int]) -> pl.DataFrame:
    """One row per player-game with xfp, fp and usage shares."""
    fo = (load("ff_opportunity", seasons)
          .with_columns(pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64))
          .select("season", "week", pl.col("player_id").alias("gsis_id"), pl.col("posteam").alias("team"),
                  _pts("_exp").alias("xfp"), _pts("").alias("fp"),
                  pl.col("rec_attempt").alias("targets"), pl.col("rec_attempt_team").alias("team_targets"),
                  pl.col("rec_air_yards").alias("air"), pl.col("rec_air_yards_team").alias("team_air"),
                  pl.col("rush_attempt").alias("carries"), pl.col("rush_attempt_team").alias("team_carries"),
                  pl.col("pass_attempt").alias("pass_att"),
                  *[e for part, terms in PARTS.items() for e in (_pts("_exp", terms).alias(f"{part}_xfp"),
                                                                 _pts("", terms).alias(f"{part}_fp"))],
                  *[e for c in DETAIL for e in (pl.col(c).alias(c), pl.col(f"{c}_exp").alias(f"{c}_exp"))])
          .drop_nulls("gsis_id").unique(["season", "week", "gsis_id"]))
    pbp = (load("pbp", seasons).filter((pl.col("season_type") == "REG") & (pl.col("yardline_100") <= 10)
                                       & (pl.col("two_point_attempt").fill_null(0) == 0))
           .with_columns(pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64)))
    team_i10 = (pbp.filter((pl.col("rush_attempt") == 1) | (pl.col("pass_attempt") == 1))
                .group_by("season", "week", pl.col("posteam").alias("team")).agg(pl.len().alias("team_i10")))
    who = pl.concat([
        pbp.filter(pl.col("rush_attempt") == 1).select("season", "week", pl.col("rusher_player_id").alias("gsis_id")),
        pbp.filter((pl.col("pass_attempt") == 1) & (pl.col("sack") == 0))
           .select("season", "week", pl.col("receiver_player_id").alias("gsis_id")),
    ]).drop_nulls().group_by("season", "week", "gsis_id").agg(pl.len().alias("i10"))
    df = (fo.join(who, on=["season", "week", "gsis_id"], how="left")
            .join(team_i10, on=["season", "week", "team"], how="left")
            .with_columns(pl.col("i10").fill_null(0)))
    share = lambda a, b: pl.when(pl.col(b) > 0).then(pl.col(a) / pl.col(b))  # noqa: E731
    return df.with_columns((pl.col("fp") - pl.col("xfp")).alias("fpoe"),
                           share("targets", "team_targets").alias("tgt_share"),
                           share("carries", "team_carries").alias("car_share"),
                           share("air", "team_air").alias("air_share"),
                           share("i10", "team_i10").alias("i10_share"))


def trailing(seasons: list[int]) -> pl.DataFrame:
    """Features for each player's *next* game: (season, week of that game, gsis_id) -> features from earlier games."""
    g = player_games(seasons).sort("season", "gsis_id", "week")
    cols = ["xfp", "fp", "fpoe", "tgt_share", "air_share", "i10_share"]
    by = ["season", "gsis_id"]
    out = g.select("season", "week", "gsis_id", *[
        e for c in cols for e in (
            pl.col(c).shift(1).over(by).alias(f"{c}_l1"),
            pl.col(c).shift(1).rolling_mean(3, min_samples=2).over(by).alias(f"{c}_l3"),
        )], pl.col("tgt_share").shift(1).cum_sum().over(by).alias("_ts_sum"),
        (pl.col("week").cum_count().over(by) - 1).alias("prior_games"))
    # target-share trend: last 2 games vs the season before them
    l2 = g.select(pl.col("tgt_share").shift(1).rolling_mean(2, min_samples=2).over(by).alias("tgt_l2"),
                  pl.col("tgt_share").shift(1).rolling_sum(2, min_samples=2).over(by).alias("_l2sum"))
    out = pl.concat([out, l2], how="horizontal").with_columns(
        pl.when(pl.col("prior_games") >= 4)
          .then(pl.col("tgt_l2") - (pl.col("_ts_sum") - pl.col("_l2sum")) / (pl.col("prior_games") - 2))
          .alias("tgt_trend")).drop("_ts_sum", "_l2sum", "tgt_l2")
    return out


def add_opportunity(df: pl.DataFrame, seasons: list[int]) -> pl.DataFrame:
    t = trailing(list(seasons))
    df = df.join(t, on=["season", "week", "gsis_id"], how="left")
    return df.with_columns((pl.col("xfp_l3") - pl.col("proj")).alias("xfp_vs_proj"))


def groups() -> list[tuple[str, pl.Expr, list[str]]]:
    def terc(col, labels):
        q1 = pl.col(col).quantile(1 / 3).over("pos")
        q2 = pl.col(col).quantile(2 / 3).over("pos")
        return (pl.when(pl.col(col).is_null()).then(pl.lit(None))
                  .when(pl.col(col) <= q1).then(pl.lit(labels[0]))
                  .when(pl.col(col) <= q2).then(pl.lit(labels[1])).otherwise(pl.lit(labels[2])))
    skill = ["QB", "RB", "WR", "TE"]
    return [
        ("pts over expected, last game", terc("fpoe_l1", ("unlucky", "neutral", "lucky")), skill),
        ("pts over expected, last 3", terc("fpoe_l3", ("unlucky", "neutral", "lucky")), skill),
        ("expected pts last 3 vs ESPN", terc("xfp_vs_proj", ("role below ESPN", "in line", "role above ESPN")), skill),
        ("target share trend", terc("tgt_trend", ("falling", "flat", "rising")), ["RB", "WR", "TE"]),
        ("air-yard share, last 3", terc("air_share_l3", ("low", "mid", "high")), ["WR", "TE"]),
        ("inside-10 share, last 3", terc("i10_share_l3", ("low", "mid", "high")), ["RB", "WR", "TE"]),
    ]
