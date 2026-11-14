from datetime import datetime, timezone

import polars as pl
import pytest

from ff_model.research.nflverse import NFLVERSE_DIR
from ff_model.sources import stadiums, weather


@pytest.mark.skipif(not (NFLVERSE_DIR / "schedules" / "2026.parquet").exists(), reason="no nflverse schedule")
def test_every_2026_stadium_is_in_the_table():
    s = pl.read_parquet(NFLVERSE_DIR / "schedules" / "2026.parquet")
    missing = sorted(set(s["stadium"].drop_nulls()) - set(stadiums.STADIUMS))
    assert missing == []


def test_weather_summary_uses_kickoff_window():
    k = datetime(2026, 11, 15, 18, 0, tzinfo=timezone.utc)
    hours = [f"2026-11-15T{h:02d}:00" for h in range(15, 24)]
    body = [{"hourly": {"time": hours,
                        "temperature_2m": [30.0 + h for h in range(9)],
                        "wind_speed_10m": [5.0, 5, 5, 12, 14, 16, 18, 30, 30],
                        "wind_gusts_10m": [9.0, 9, 9, 20, 25, 28, 31, 50, 50],
                        "precipitation": [0.0] * 9, "precipitation_probability": [10] * 9}}]
    s = weather.summarize(body, [{"game_id": 1, "kickoff": k}])[0]
    # kickoff 18:00 .. 21:00 -> indices 3..6
    assert s["wind_mph"] == pytest.approx((12 + 14 + 16 + 18) / 4)
    assert s["gust_mph"] == 31
    assert s["temp_f"] == pytest.approx(30 + (3 + 4 + 5 + 6) / 4)


@pytest.mark.skipif(not (NFLVERSE_DIR / "schedules" / "2026.parquet").exists(), reason="no nflverse schedule")
def test_every_kickoff_slot_finds_its_games_with_stadiums():
    """At each kickoff-relative slot (Thursday, international, Saturday, ...), the context step must
    find every game that slot is the last chance for, with a stadium and roof."""
    from ff_model.archive.context import upcoming_games
    from ff_model.archive.slots import all_slots
    from ff_model.config import DB_PATH
    from ff_model.db import connect
    if not DB_PATH.exists():
        pytest.skip("no archive")
    con = connect()
    checked = 0
    for s in all_slots(con):
        if s.last_kick is None or s.at < datetime.now(timezone.utc):
            continue
        games = upcoming_games(con, s.week, s.at)
        covered = {g["kickoff_utc"] for g in games}
        due = {r["kickoff_utc"] for r in con.execute(
            "SELECT kickoff_utc FROM pro_games WHERE week=? AND kickoff_utc > ? AND kickoff_utc <= ?",
            (s.week, s.at.astimezone(timezone.utc).isoformat(), s.last_kick.astimezone(timezone.utc).isoformat()))}
        assert due, (s.key, "slot serves no games: window query is wrong")
        assert due <= covered, (s.key, due - covered)
        assert all(g["roof"] in ("outdoors", "closed", "dome") for g in games)
        checked += 1
    assert checked > 0


@pytest.mark.skipif(not (NFLVERSE_DIR / "rosters_weekly" / "2026.parquet").exists(), reason="no nflverse 2026")
def test_factor_frame_uses_requested_seasons():
    """Regression: factor_frame once read 2019-2025 regardless of `seasons`, leaving 2026 unjoined."""
    from ff_model.research.espn_history import HIST_DIR
    from ff_model.research.factor_tests import factor_frame
    if not (HIST_DIR / "2026").exists():
        pytest.skip("no 2026 projections")
    df = factor_frame([2026], depth=1.0, with_u=False)
    assert df["team"].is_not_null().mean() > 0.95
    assert df["opp"].is_not_null().mean() > 0.95
