"""League constants, paths, and credential loading.

Credentials come from the gitignored .env and must never be printed or logged.
"""
from __future__ import annotations

import os
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "logs"
DB_PATH = DATA_DIR / "archive.db"

SEASON = 2026
LEAGUE_ID = 765545014
MY_TEAM_ID = 9  # Nostrildamus

ET = ZoneInfo("America/New_York")

ESPN_LEAGUE_URL = (
    f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{SEASON}"
    f"/segments/0/leagues/{LEAGUE_ID}"
)
ESPN_SEASON_URL = f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{SEASON}"

# The Odds API free tier: 500 credits/month, assumed to reset on the 1st (UTC).
# Cost = #markets x #regions per call: game lines (all games, one call) = 2;
# player props (one call per game) = 1 per market. /events is free.
ODDS_BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
ODDS_MARKETS = "spreads,totals"
ODDS_REGIONS = "us"
ODDS_CREDITS_PER_CALL = 2
ODDS_REMAINING_FLOOR = 40  # never spend the last credits; leave room for manual use
ODDS_MIN_GAP_MINUTES = 30  # don't refetch game lines more often than this
# Highest value first; the budget planner drops markets from the end when short.
PROP_MARKETS = ["player_anytime_td", "player_receptions", "player_reception_yds",
                "player_rush_yds", "player_pass_yds"]


def _env() -> dict[str, str]:
    vals = dotenv_values(ROOT / ".env")
    return {k: v for k, v in {**vals, **os.environ}.items() if v is not None}


def espn_cookies() -> dict[str, str]:
    env = _env()
    try:
        return {"espn_s2": env["espn_s2"], "SWID": env["SWID"]}
    except KeyError as e:
        raise RuntimeError(f"missing {e.args[0]} in .env") from None


def odds_api_key() -> str | None:
    return _env().get("ODDS_API_KEY") or None
