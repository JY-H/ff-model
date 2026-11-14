"""League constants, paths, and credential loading.

Credentials and league/team ids come from the gitignored .env. Credentials
must never be printed or logged.
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
ET = ZoneInfo("America/New_York")


def _env() -> dict[str, str]:
    vals = dotenv_values(ROOT / ".env")
    return {k: v for k, v in {**vals, **os.environ}.items() if v is not None}


def _required(name: str) -> str:
    try:
        return _env()[name]
    except KeyError:
        raise RuntimeError(f"missing {name} in .env") from None


LEAGUE_ID = int(_required("ESPN_LEAGUE_ID"))
MY_TEAM_ID = int(_required("ESPN_TEAM_ID"))

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


def espn_cookies() -> dict[str, str]:
    return {"espn_s2": _required("espn_s2"), "SWID": _required("SWID")}


def odds_api_key() -> str | None:
    return _env().get("ODDS_API_KEY") or None
