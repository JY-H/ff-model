"""Scoring tests. The data tests compare against ESPN's own league-scored
numbers in the local archive and skip when it isn't present."""
import json
import sqlite3

import polars as pl
import pytest

from ff_model.config import DB_PATH
from ff_model.research import scoring

RULES = {3: (0.04, 0.04), 4: (4.0, 4.0), 53: (1.0, 1.0), 85: (-1.0, -1.0), 99: (0.0, 1.0),
         91: (0.0, 3.0), 123: (0.0, -1.0), 130: (0.0, 2.0), 132: (0.0, -1.0)}

needs_archive = pytest.mark.skipif(not DB_PATH.exists(), reason="no local archive")


def test_dst_tiers_sack_sign_and_non_offense_points():
    team = pl.DataFrame({
        "season": [2026, 2026], "week": [1, 1], "season_type": ["REG", "REG"],
        "team": ["AAA", "BBB"], "opponent_team": ["BBB", "AAA"],
        "passing_yards": [250, 300], "sack_yards_lost": [-20, 15], "rushing_yards": [60, 80],
        "def_sacks": [3.0, 1.0], "def_interceptions": [0, 0], "fumble_recovery_opp": [0, 0],
        "def_fg_blocks": [0, 0], "def_punt_blocks": [0, 0], "def_pat_blocks": [0, 0],
        "def_safeties": [0, 0], "def_tds": [1, 0], "fumble_recovery_tds": [0, 0], "special_teams_tds": [0, 0],
    })
    sched = pl.DataFrame({"season": [2026], "week": [1], "game_type": ["REG"], "home_team": ["AAA"],
                          "away_team": ["BBB"], "home_score": [31], "away_score": [10]})
    d = scoring.score_dst(scoring.dst_inputs(team, sched), RULES).sort("team")
    # BBB allowed 31, but 6 came from AAA's pick-six -> 25 (no tier); yards 250-20+60 = 290 -> 200-299 tier
    bbb = d.filter(pl.col("team") == "BBB").row(0, named=True)
    assert bbb["points_allowed"] == 25 and bbb["yards_allowed"] == 290
    assert bbb["fpts"] == pytest.approx(1.0 + 2.0)
    # AAA allowed 10 (7-13 tier: 3), yards 300-15+80 = 365 (350-399: -1), 3 sacks
    aaa = d.filter(pl.col("team") == "AAA").row(0, named=True)
    assert aaa["fpts"] == pytest.approx(3.0 + 3.0 - 1.0)


@needs_archive
def test_rescoring_espn_projections_matches_espn():
    con = sqlite3.connect(DB_PATH)
    rows = con.execute("SELECT position_id, proj_points, proj_stats_json FROM player_snapshots"
                       " WHERE proj_stats_json IS NOT NULL LIMIT 2000").fetchall()
    if not rows:
        pytest.skip("no projections archived")
    df = pl.DataFrame([{"position_id": r[0], "espn": r[1], "stats": json.loads(r[2])} for r in rows],
                      schema_overrides={"stats": pl.Object})
    out = scoring.score_espn_statlines(df, scoring.league_rules())
    assert (out["fpts"] - out["espn"]).abs().max() < 0.05
