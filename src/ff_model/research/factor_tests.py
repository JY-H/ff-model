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


def _team_games(seasons=SEASONS) -> pl.DataFrame:
    """One row per team-game: opponent, home/away, weather, lines, implied total."""
    s = load("schedules", list(seasons)).filter(pl.col("game_type") == "REG")
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


def factor_frame(seasons=SEASONS, depth: float = 1.0, with_u: bool = True) -> pl.DataFrame:
    con = connect()
    rules = scoring.league_rules(con)
    seasons = list(seasons)
    df = relevant(build(seasons), depth=depth)

    # Player's NFL team that week. ESPN's historical entries don't carry it (proTeamId is 0, and
    # the player-level team is *current*), so use nflverse weekly rosters; D/ST ids encode the team.
    ids = (load("ff_playerids").select(pl.col("espn_id").cast(pl.Int64, strict=False), "gsis_id")
           .drop_nulls().unique("espn_id"))
    rost = (load("rosters_weekly", seasons)
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
    g = _team_games(seasons).with_columns(pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64))
    df = df.join(g, on=["season", "week", "team"], how="left")

    # --- offense pace and red-zone stalls (team, prior weeks) ---
    ts = load("team_stats", seasons).filter(pl.col("season_type") == "REG").with_columns(
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
    ps = scoring.score_players(load("player_stats", seasons).filter(pl.col("season_type") == "REG"), rules)
    fpa = (ps.filter(pl.col("position").is_in(["QB", "RB", "WR", "TE", "K"]))
             .group_by("season", "week", "opponent_team", "position").agg(pl.col("fpts").sum().alias("fpa")))
    fpa_t = pl.concat([
        _trailing(fpa.filter(pl.col("position") == p).rename({"opponent_team": "defense"}), "defense", ["fpa"])
        .with_columns(pl.lit(p).alias("pos")) for p in ["QB", "RB", "WR", "TE", "K"]])
    df = df.join(fpa_t.select("season", "week", pl.col("defense").alias("opp"), "pos", "fpa_pg"),
                 on=["season", "week", "opp", "pos"], how="left")

    # --- injury designation that week (players who were listed) ---
    inj = (load("injuries", seasons).filter(pl.col("game_type") == "REG")
           .select(pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64), "gsis_id", "report_status",
                   "practice_status", pl.col("report_primary_injury").alias("injury"))
           .unique(["season", "week", "gsis_id"]))
    df = (df.join(inj, on=["season", "week", "gsis_id"], how="left")
            .with_columns(pl.col("report_status").fill_null("none")))

    df = _add_return_from_absence(df, seasons)
    df = _add_mismatch(df, seasons)
    df = _add_line_injuries(df, seasons)
    df = _add_vacated(df, seasons)

    # --- where the actual landed in the player's predicted range ---
    if with_u:
        m = Marginals.load(MODEL_PATH)
        df = df.with_columns(pl.Series("u", cdf(m.quantiles(df), df["actual"].to_numpy())))
    return df


def _add_return_from_absence(df: pl.DataFrame, seasons=SEASONS) -> pl.DataFrame:
    """games_missed: consecutive games missed right before this one (roster status INA/RES, byes skipped).
    games_back: 1 = first game back after missing 2+ games, 2 = second, ...; 0 otherwise.
    from_ir: the absence included a week on reserve (IR/PUP)."""
    r = (load("rosters_weekly", list(seasons)).filter(pl.col("game_type") == "REG")
         .select(pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64), "gsis_id", "status")
         .drop_nulls("gsis_id").unique(["season", "week", "gsis_id"]).sort("gsis_id", "season", "week"))
    rows = []
    for (gid, season), g in r.group_by(["gsis_id", "season"], maintain_order=True):
        streak, ir, back, last_absence_ir = 0, False, 0, False
        for week, status in zip(g["week"], g["status"]):
            out = status in ("INA", "RES")
            if out:
                streak, ir = streak + 1, ir or status == "RES"
                back = 0
                continue
            if streak >= 2:
                back, last_absence_ir = 1, ir
            elif back:
                back += 1
            rows.append((gid, season, week, streak, back if back <= 3 else 0, last_absence_ir and back > 0))
            streak, ir = 0, False
    ra = pl.DataFrame(rows, schema=["gsis_id", "season", "week", "games_missed", "games_back", "from_ir"], orient="row")
    return df.join(ra, on=["gsis_id", "season", "week"], how="left")


def _add_line_injuries(df: pl.DataFrame, seasons=SEASONS, line=None) -> pl.DataFrame:
    """Own OL starters out, opponent OL and DB starters out (injury-report based)."""
    from .line_injuries import starters_out
    so = line if line is not None else starters_out(list(seasons))
    own = so.select("season", "week", "team", "ol_out")
    opp = so.select("season", "week", pl.col("team").alias("opp"), pl.col("ol_out").alias("opp_ol_out"),
                    pl.col("db_out").alias("opp_db_out"))
    return (df.join(own, on=["season", "week", "team"], how="left")
              .join(opp, on=["season", "week", "opp"], how="left"))


def _add_vacated(df: pl.DataFrame, seasons=SEASONS, vac=None) -> pl.DataFrame:
    """Share of the team's targets / carries held by key teammates who are out (injury-report based)."""
    from .vacated import vacated
    v = vac if vac is not None else vacated(list(seasons))
    return (df.join(v.select("season", "week", "team", "vacated_tgt", "vacated_car"), on=["season", "week", "team"], how="left")
              .with_columns(pl.col("vacated_tgt").fill_null(0.0), pl.col("vacated_car").fill_null(0.0)))


def _add_mismatch(df: pl.DataFrame, seasons=SEASONS) -> pl.DataFrame:
    """Opponent's implied total (market view), and the stats view people quote:
    this team's points allowed per game + opponent's points scored per game (prior weeks)."""
    sch = load("schedules", list(seasons)).filter(pl.col("game_type") == "REG")
    games = pl.concat([
        sch.select("season", "week", pl.col("home_team").alias("team"), pl.col("home_score").alias("pf"),
                   pl.col("away_score").alias("pa")),
        sch.select("season", "week", pl.col("away_team").alias("team"), pl.col("away_score").alias("pf"),
                   pl.col("home_score").alias("pa")),
    ]).drop_nulls("pf").with_columns(pl.col("pf").cast(pl.Float64), pl.col("pa").cast(pl.Float64))
    t = _trailing(games, "team", ["pf", "pa"])
    own = t.select("season", "week", "team", pl.col("pa_pg").alias("own_pa_pg"), pl.col("n_prior").alias("own_np"))
    opp = t.select("season", "week", pl.col("team").alias("opp"), pl.col("pf_pg").alias("opp_pf_pg"),
                   pl.col("n_prior").alias("opp_np"))
    return (df.join(own, on=["season", "week", "team"], how="left").join(opp, on=["season", "week", "opp"], how="left")
              .with_columns((pl.col("total_line") - pl.col("implied_total")).alias("opp_implied_total"),
                            pl.when((pl.col("own_np") >= MIN_PRIOR_GAMES) & (pl.col("opp_np") >= MIN_PRIOR_GAMES))
                            .then(pl.col("own_pa_pg") + pl.col("opp_pf_pg")).alias("mismatch")))


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

    def cnt(col):
        return (pl.when(pl.col(col).is_null()).then(pl.lit(None)).when(pl.col(col) == 0).then(pl.lit("0 out"))
                  .when(pl.col(col) == 1).then(pl.lit("1 out")).otherwise(pl.lit("2+ out")))

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
        # among players who played: final practice status of the week
        ("practice (played)", pl.when(~pl.col("played")).then(pl.lit(None))
                                .when(pl.col("practice_status").str.contains("Did Not")).then(pl.lit("DNP"))
                                .when(pl.col("practice_status").str.contains("Limited")).then(pl.lit("limited"))
                                .when(pl.col("practice_status").str.contains("Full")).then(pl.lit("full (listed)"))
                                .when(pl.col("report_status") == "none").then(pl.lit("not on report")), skill),
        ("hamstring (played, listed)", pl.when(~pl.col("played") | pl.col("practice_status").is_null()).then(pl.lit(None))
                                         .when(pl.col("injury").str.contains("(?i)hamstring")).then(pl.lit("hamstring"))
                                         .otherwise(pl.lit("other injury")), skill),
        ("return after 2+ missed", pl.when(pl.col("games_back") == 1).then(pl.lit("1st game back"))
                                     .when(pl.col("games_back") == 2).then(pl.lit("2nd game back"))
                                     .when(pl.col("games_back") == 3).then(pl.lit("3rd game back"))
                                     .when(pl.col("games_back") == 0).then(pl.lit("not returning")), skill),
        ("return from IR", pl.when((pl.col("games_back") >= 1) & pl.col("from_ir")).then(pl.lit("games 1-3 after IR"))
                             .when(pl.col("games_back") == 0).then(pl.lit("not returning")), skill),
        ("own OL starters out", cnt("ol_out"), skill),
        ("opp OL starters out", cnt("opp_ol_out"), ["D/ST", "QB", "RB"]),
        ("opp DB starters out", cnt("opp_db_out"), ["QB", "WR", "TE"]),
        ("teammate targets vacated", pl.when(pl.col("vacated_tgt") >= 0.20).then(pl.lit("20%+"))
                                       .when(pl.col("vacated_tgt") > 0).then(pl.lit("under 20%")).otherwise(pl.lit("none")),
         ["WR", "TE", "RB"]),
        ("teammate carries vacated", pl.when(pl.col("vacated_car") >= 0.40).then(pl.lit("40%+"))
                                       .when(pl.col("vacated_car") > 0).then(pl.lit("under 40%")).otherwise(pl.lit("none")), ["RB"]),
        ("opp implied total", terc("opp_implied_total", pl.lit(True)), skill),
        ("D-vs-O mismatch (stats)", terc("mismatch", pl.col("mismatch").is_not_null(),
                                          ("well matched", "avg", "badly matched")), skill),
    ]


def screen(df: pl.DataFrame, groups=None) -> pl.DataFrame:
    rows = []
    for name, grp, positions in (groups if groups is not None else _groups(df)):
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



def run_screen():
    """Regenerate data/factor_screen.csv (historical evidence shown on the page)."""
    from ..config import DATA_DIR
    df = factor_frame()
    r = screen(df)
    r.write_csv(DATA_DIR / "factor_screen.csv")
    return r
