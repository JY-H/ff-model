"""Pre-game context archived with each snapshot: kickoff-window weather and the
week's injury report. Both are perishable as-of-lock information."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

import polars as pl

from ..config import SEASON
from ..db import iso, load_raw, save_raw, utcnow
from ..research import nflverse
from ..sources import stadiums, weather

log = logging.getLogger("ff_model")

ABBR_FIX = {"WSH": "WAS", "LAR": "LA"}  # ESPN -> nflverse
NFLVERSE_LIVE = ["injuries", "rosters_weekly", "snap_counts", "schedules"]
WEATHER_HORIZON = timedelta(days=8)


def espn_abbr(con) -> dict[int, str]:
    row = con.execute("SELECT * FROM raw_responses WHERE kind='pro_schedule' AND status_code=200"
                      " ORDER BY id DESC LIMIT 1").fetchone()
    return {t["id"]: ABBR_FIX.get(t["abbrev"].upper(), t["abbrev"].upper())
            for t in load_raw(row)["settings"]["proTeams"]}


def upcoming_games(con, week: int, now: datetime) -> list[dict]:
    """This week's games that haven't kicked off, with stadium and roof from nflverse + our table."""
    abbr = espn_abbr(con)
    sch = nflverse.load("schedules", [SEASON]).filter(pl.col("week") == week)
    stadium_of = {(r["home_team"], r["away_team"]): r["stadium"] for r in sch.iter_rows(named=True)}
    games = []
    for r in con.execute("SELECT * FROM pro_games WHERE season=? AND week=? AND kickoff_utc > ?",
                         (SEASON, week, iso(now))):
        kick = datetime.fromisoformat(r["kickoff_utc"])
        if kick > now + WEATHER_HORIZON:
            continue
        home, away = abbr.get(r["home_pro_team_id"]), abbr.get(r["away_pro_team_id"])
        stadium = stadium_of.get((home, away))
        if stadium not in stadiums.STADIUMS:
            log.warning("no stadium for %s@%s (%s)", away, home, stadium)
            continue
        lat, lon, _ = stadiums.STADIUMS[stadium]
        games.append({"game_id": r["game_id"], "week": week, "kickoff_utc": r["kickoff_utc"], "home": home,
                      "away": away, "stadium": stadium, "roof": stadiums.model_roof(stadium), "lat": lat, "lon": lon})
    return games


def archive_context(con, sid: int, week: int) -> list[str]:
    """Fetch and archive weather + injury report for this snapshot. Returns error strings."""
    errors = []
    try:
        nflverse.pull([SEASON], datasets=NFLVERSE_LIVE)
    except Exception as e:
        log.error("nflverse refresh failed: %s", e)
        errors.append(f"nflverse: {type(e).__name__}")
    try:
        inj = nflverse.load("injuries", [SEASON]).filter(pl.col("week") == week).drop("_pulled_at", strict=False)
        save_raw(con, sid, "nflverse", "nflverse_injuries", "nflreadpy.load_injuries",
                 {"season": SEASON, "week": week}, 200, inj.to_dicts())
    except Exception as e:
        log.error("injury archive failed: %s", e)
        errors.append(f"injuries: {type(e).__name__}")
    try:
        games = upcoming_games(con, week, utcnow())
        outdoor = [dict(g, kickoff=datetime.fromisoformat(g["kickoff_utc"])) for g in games if g["roof"] == "outdoors"]
        res = weather.fetch(outdoor) if outdoor else None
        params = {"games": games}
        if res:
            url, p, status, body = res
            save_raw(con, sid, "weather", "weather", url, {**p, "games": games}, status, body)
        elif games:  # all domes: still record stadium/roof
            save_raw(con, sid, "weather", "weather", weather.URL, params, 200, None)
    except Exception as e:
        log.error("weather failed: %s", e)
        errors.append(f"weather: {type(e).__name__}")
    return errors
