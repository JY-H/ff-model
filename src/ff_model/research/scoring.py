"""This league's scoring, applied to nflverse actuals and ESPN projected stat lines.

Rules are expressed in ESPN stat ids (taken from the archived league settings).
nflverse columns are mapped into those ids, so one scorer covers both sources.
"""
from __future__ import annotations

import polars as pl

from ..config import LEAGUE_ID
from ..db import connect, load_raw

DST_SLOT = "16"

# ESPN stat id -> nflverse player_stats expression (offense + kickers)
PLAYER_MAP: dict[int, pl.Expr] = {
    3: pl.col("passing_yards"),
    4: pl.col("passing_tds"),
    19: pl.col("passing_2pt_conversions"),
    20: pl.col("passing_interceptions"),
    24: pl.col("rushing_yards"),
    25: pl.col("rushing_tds"),
    26: pl.col("rushing_2pt_conversions"),
    42: pl.col("receiving_yards"),
    43: pl.col("receiving_tds"),
    44: pl.col("receiving_2pt_conversions"),
    53: pl.col("receptions"),
    63: pl.col("fumble_recovery_tds"),
    72: pl.col("rushing_fumbles_lost") + pl.col("receiving_fumbles_lost") + pl.col("sack_fumbles_lost"),
    101: pl.col("special_teams_tds"),  # nflverse doesn't split KR/PR TDs; both score 6
    80: pl.col("fg_made_0_19") + pl.col("fg_made_20_29") + pl.col("fg_made_30_39"),
    77: pl.col("fg_made_40_49"),
    198: pl.col("fg_made_50_59"),  # not in espn-api's PLAYER_STATS_MAP; 5 pts in this league
    201: pl.col("fg_made_60_"),
    85: pl.col("fg_missed") + pl.col("fg_blocked"),  # ESPN counts a blocked FG as a miss
    86: pl.col("pat_made"),
}

# Points-allowed and yards-allowed tiers: stat id -> (low, high) inclusive
PA_TIERS = {89: (0, 0), 90: (1, 6), 91: (7, 13), 92: (14, 17), 123: (28, 34), 124: (35, 45), 125: (46, 999)}
YA_TIERS = {128: (-999, 99), 129: (100, 199), 130: (200, 299), 132: (350, 399),
            133: (400, 449), 134: (450, 499), 135: (500, 549), 136: (550, 9999)}


def league_rules(con=None) -> dict[int, tuple[float, float]]:
    """stat id -> (points for players, points for D/ST)."""
    con = con or connect()
    row = con.execute("SELECT * FROM raw_responses WHERE kind='settings' AND status_code=200 AND league_id=?"
                      " ORDER BY id DESC LIMIT 1", (LEAGUE_ID,)).fetchone()
    items = load_raw(row)["settings"]["scoringSettings"]["scoringItems"]
    return {it["statId"]: (it["points"], (it.get("pointsOverrides") or {}).get(DST_SLOT, it["points"]))
            for it in items}


def _score(df: pl.DataFrame, rules, dst: bool) -> pl.Expr:
    terms = [pl.col(f"st_{sid}").fill_null(0) * pts[1 if dst else 0]
             for sid, pts in rules.items() if f"st_{sid}" in df.columns and pts[1 if dst else 0]]
    return pl.sum_horizontal(terms) if terms else pl.lit(0.0)


def score_players(ps: pl.DataFrame, rules) -> pl.DataFrame:
    """nflverse player_stats -> adds st_* columns and fpts."""
    df = ps.with_columns([e.fill_null(0).alias(f"st_{sid}") for sid, e in PLAYER_MAP.items()])
    return df.with_columns(_score(df, rules, dst=False).alias("fpts"))


def dst_inputs(team_stats: pl.DataFrame, schedules: pl.DataFrame) -> pl.DataFrame:
    """One row per team-game with the defense's raw stats and what it allowed."""
    ts = team_stats.filter(pl.col("season_type") == "REG")
    # Net yards: sack yardage is stored negative in recent nflverse seasons; abs() guards older ones.
    # ESPN's points allowed excludes points the opponent's defense/special teams scored.
    opp = ts.select("season", "week", pl.col("team").alias("opponent_team"),
                    (pl.col("passing_yards") - pl.col("sack_yards_lost").abs() + pl.col("rushing_yards")).alias("yards_allowed"),
                    (6 * (pl.col("def_tds") + pl.col("fumble_recovery_tds") + pl.col("special_teams_tds")).fill_null(0)
                     + 2 * pl.col("def_safeties").fill_null(0)).alias("opp_non_offense_pts"))
    sch = schedules.filter(pl.col("game_type") == "REG")
    pts = pl.concat([
        sch.select("season", "week", pl.col("home_team").alias("team"), pl.col("away_score").alias("points_allowed")),
        sch.select("season", "week", pl.col("away_team").alias("team"), pl.col("home_score").alias("points_allowed")),
    ])
    return (ts.join(opp, on=["season", "week", "opponent_team"], how="left")
              .join(pts, on=["season", "week", "team"], how="left")
              .with_columns((pl.col("points_allowed") - pl.col("opp_non_offense_pts")).alias("points_allowed")))


def score_dst(d: pl.DataFrame, rules) -> pl.DataFrame:
    cols = {
        99: pl.col("def_sacks"), 95: pl.col("def_interceptions"), 96: pl.col("fumble_recovery_opp"),
        97: pl.col("def_fg_blocks") + pl.col("def_punt_blocks") + pl.col("def_pat_blocks"),
        98: pl.col("def_safeties"),
        103: pl.col("def_tds"),              # interception return TDs
        104: pl.col("fumble_recovery_tds"),  # fumble return TDs
        101: pl.col("special_teams_tds"),    # KR/PR/blocked-kick TDs (all 6)
    }
    cols |= {sid: pl.col("points_allowed").is_between(lo, hi).cast(pl.Int8) for sid, (lo, hi) in PA_TIERS.items()}
    cols |= {sid: pl.col("yards_allowed").is_between(lo, hi).cast(pl.Int8) for sid, (lo, hi) in YA_TIERS.items()}
    df = d.with_columns([e.fill_null(0).alias(f"st_{sid}") for sid, e in cols.items()])
    return df.with_columns(_score(df, rules, dst=True).alias("fpts"))


def score_espn_statlines(df: pl.DataFrame, rules, stats_col: str = "stats",
                         position_col: str = "position_id") -> pl.DataFrame:
    """ESPN stat lines (dict of stat id -> value, e.g. projections) -> adds fpts."""
    ids = sorted(rules)
    df = df.with_columns([pl.col(stats_col).map_elements(lambda d, k=str(sid): (d or {}).get(k, 0.0),
                                                         return_dtype=pl.Float64).alias(f"st_{sid}")
                          for sid in ids])
    is_dst = pl.col(position_col) == 16
    return df.with_columns(pl.when(is_dst).then(_score(df, rules, dst=True))
                           .otherwise(_score(df, rules, dst=False)).alias("fpts"))


def score_wide(df: pl.DataFrame, rules, position_col: str = "position_id") -> pl.DataFrame:
    """Frame with st_<espn stat id> columns (e.g. espn_history.parse()) -> adds fpts."""
    is_dst = pl.col(position_col) == 16
    return df.with_columns(pl.when(is_dst).then(_score(df, rules, dst=True))
                           .otherwise(_score(df, rules, dst=False)).alias("fpts"))
