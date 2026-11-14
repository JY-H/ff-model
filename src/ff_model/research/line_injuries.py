"""Offensive-line and secondary starters missing, per team-week, from pre-game information.

A starter is an OL (or CB/S) who played >= 60% of snaps on average in the team's
prior games that season (at least 2 games). He counts as out if this week's
injury report lists him Out or Doubtful, or the weekly roster has him on
reserve (IR/PUP). That's what is known before inactives. `*_out_actual` counts
starters who logged no snaps, for checking the report-based count.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from .nflverse import load

OL = ["T", "G", "C", "OL", "OT", "OG"]
DB = ["CB", "S", "FS", "SS", "DB"]


def _norm(name: pl.Expr) -> pl.Expr:
    """lowercase, letters only, without Jr/Sr/II/III suffixes."""
    return (name.str.to_lowercase().str.replace_all(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "")
                .str.replace_all(r"[^a-z]", ""))


def starters_out(seasons: list[int], extra_injuries: pl.DataFrame | None = None,
                 asof_week: int | None = None) -> pl.DataFrame:
    """One row per (season, week, team): ol_out, db_out (report-based) and *_out_actual.

    Live: pass this week's archived report as extra_injuries and asof_week, so the
    current week uses only what was archived at the snapshot, not later nflverse data."""
    sc = (load("snap_counts", seasons).filter(pl.col("game_type") == "REG")
          .with_columns(pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64)))
    # Snap counts only carry PFR ids, and nflverse has no PFR id for offensive linemen,
    # so link to gsis by normalized name within team-season (falling back from the PFR id).
    ros_all = load("rosters_weekly", seasons).with_columns(pl.col("season").cast(pl.Int64))
    by_pfr = ros_all.select("pfr_id", "gsis_id").drop_nulls().unique("pfr_id")
    by_name = (ros_all.select("season", "team", _norm(pl.col("full_name")).alias("nk"), "gsis_id")
               .drop_nulls().unique(["season", "team", "nk"]))
    sc = (sc.with_columns(_norm(pl.col("player")).alias("nk"))
            .join(by_pfr, left_on="pfr_player_id", right_on="pfr_id", how="left")
            .join(by_name.rename({"gsis_id": "gsis_by_name"}), on=["season", "team", "nk"], how="left")
            .with_columns(pl.coalesce("gsis_id", "gsis_by_name").alias("gsis_id")))
    inj = (load("injuries", seasons).filter(pl.col("game_type") == "REG")
           .select(pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64), "gsis_id", "report_status"))
    if asof_week is not None:
        inj = inj.filter(pl.col("week") < asof_week)
    if extra_injuries is not None:  # live: this week's archived report
        inj = pl.concat([inj, extra_injuries.select(inj.columns)], how="vertical_relaxed")
    reported_out = {(r[0], r[1], r[2]) for r in inj.filter(pl.col("report_status").is_in(["Out", "Doubtful"]))
                    .select("season", "week", "gsis_id").iter_rows()}
    ros = (load("rosters_weekly", seasons).filter(pl.col("game_type") == "REG")
           .select(pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64), "gsis_id", "status"))
    on_reserve = {(r[0], r[1], r[2]) for r in ros.filter(pl.col("status") == "RES")
                  .select("season", "week", "gsis_id").iter_rows()}

    rows = []
    for unit, positions, pct in (("ol", OL, "offense_pct"), ("db", DB, "defense_pct")):
        s = sc.filter(pl.col("position").is_in(positions))
        for (season, team), ts in s.group_by(["season", "team"]):
            hist: dict[str, list[float]] = {}
            gsis: dict[str, str] = {}
            weeks = sorted(sc.filter((pl.col("season") == season) & (pl.col("team") == team))["week"].unique())
            # live: also emit the target week, even if it follows a bye
            extra = [asof_week] if asof_week is not None and asof_week not in weeks else []
            for w in sorted(set(weeks) | set(extra)):
                st = [p for p, v in hist.items() if len(v) >= 2 and np.mean(v) >= 0.6]
                this = ts.filter(pl.col("week") == w)
                played = set(this.filter(pl.col(pct) > 0)["pfr_player_id"].to_list())
                out_rep = sum((season, w, gsis.get(p)) in reported_out or (season, w, gsis.get(p)) in on_reserve
                              for p in st)
                rows.append((unit, season, w, team, len(st), out_rep,
                             sum(p not in played for p in st) if w in weeks else None))
                for p, g, v in zip(this["pfr_player_id"], this["gsis_id"], this[pct]):
                    hist.setdefault(p, []).append(v)
                    gsis[p] = g
    df = pl.DataFrame(rows, schema=["unit", "season", "week", "team", "starters", "out", "out_actual"], orient="row")
    wide = df.pivot(on="unit", index=["season", "week", "team"], values=["starters", "out", "out_actual"])
    return wide.rename({"starters_ol": "ol_starters", "starters_db": "db_starters", "out_ol": "ol_out",
                        "out_db": "db_out", "out_actual_ol": "ol_out_actual", "out_actual_db": "db_out_actual"})
