"""Usage vacated by injured teammates, per team-week, from pre-game information.

A key player has >= 15% of the team's targets (or >= 30% of its carries) in the
team's prior games that season (at least 2 games). He counts as out if this
week's injury report lists him Out/Doubtful or the weekly roster has him on
reserve (IR). vacated_tgt / vacated_car = summed prior shares of key players out.
"""
from __future__ import annotations

import polars as pl

from .nflverse import load

TGT_KEY, CAR_KEY = 0.15, 0.30


def vacated(seasons: list[int], extra_injuries: pl.DataFrame | None = None, asof_week: int | None = None) -> pl.DataFrame:
    ps = (load("player_stats", seasons).filter(pl.col("season_type") == "REG")
          .select(pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64), "player_id", "player_display_name",
                  "position", "team", pl.col("targets").fill_null(0), pl.col("carries").fill_null(0)))
    team = ps.group_by("season", "week", "team").agg(pl.col("targets").sum().alias("t_tgt"), pl.col("carries").sum().alias("t_car"))
    ps = ps.join(team, on=["season", "week", "team"])
    inj = (load("injuries", seasons).filter(pl.col("game_type") == "REG")
           .select(pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64), pl.col("gsis_id").alias("player_id"), "report_status"))
    if asof_week is not None:
        inj = inj.filter(pl.col("week") < asof_week)
    if extra_injuries is not None:
        inj = pl.concat([inj, extra_injuries.rename({"gsis_id": "player_id"}).select(inj.columns)], how="vertical_relaxed")
    out_rep = inj.filter(pl.col("report_status").is_in(["Out", "Doubtful"])).select("season", "week", "player_id").unique()
    ros = (load("rosters_weekly", seasons).filter(pl.col("game_type") == "REG")
           .select(pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64), pl.col("gsis_id").alias("player_id"), "status", "team"))
    res = ros.filter(pl.col("status") == "RES").select("season", "week", "player_id").unique()
    out = pl.concat([out_rep, res]).unique()

    rows = []
    for (season, tm), g in ps.sort("week").group_by(["season", "team"], maintain_order=True):
        weeks = sorted(g["week"].unique().to_list())
        target_weeks = sorted(set(weeks) | ({asof_week} if asof_week else set()))
        for w in target_weeks:
            prior = g.filter(pl.col("week") < w)
            if prior["week"].n_unique() < 2:
                continue
            tot = prior.group_by("player_id", "player_display_name", "position").agg(
                pl.col("targets").sum(), pl.col("carries").sum())
            tt = prior.unique(["week"])["t_tgt"].sum()
            tc = prior.unique(["week"])["t_car"].sum()
            tot = tot.with_columns((pl.col("targets") / max(tt, 1)).alias("tshare"), (pl.col("carries") / max(tc, 1)).alias("cshare"))
            key = tot.filter((pl.col("tshare") >= TGT_KEY) | (pl.col("cshare") >= CAR_KEY))
            outs = out.filter((pl.col("season") == season) & (pl.col("week") == w))["player_id"].to_list()
            k_out = key.filter(pl.col("player_id").is_in(outs))
            rows.append((season, w, tm, float(k_out["tshare"].sum()), float(k_out["cshare"].sum()),
                         "; ".join(f"{r['player_display_name']} ({r['position']}, {r['tshare']:.0%} of targets, {r['cshare']:.0%} of carries)"
                                   for r in k_out.iter_rows(named=True)),
                         ",".join(k_out["player_id"].to_list())))
    return pl.DataFrame(rows, schema=["season", "week", "team", "vacated_tgt", "vacated_car", "vacated_who", "vacated_ids"], orient="row")
