import pytest

from ff_model.config import DB_PATH

pytestmark = pytest.mark.skipif(not DB_PATH.exists(), reason="needs the local archive")


def test_review_adds_up():
    """Final score = ESPN projection + role + luck + K/DST miss, for both sides."""
    from ff_model.model.review import week_review
    r = week_review(4)
    for side in ("me", "opp"):
        t = r["totals"][side]
        assert t["proj"] + t["role"] + t["luck"] + t["kdst"] == pytest.approx(t["actual"], abs=0.01)
    for p in r["players"]["me"]:
        if p["xfp"] is not None:
            assert p["proj"] + p["role"] + p["luck"] == pytest.approx(p["actual"])


def test_opportunity_features_use_earlier_games_only():
    from ff_model.research.opportunity import player_games, trailing
    import polars as pl
    g = player_games([2024]).sort("gsis_id", "week")
    t = trailing([2024])
    pid = g.group_by("gsis_id").len().sort("len", descending=True)["gsis_id"][0]
    gp, tp = g.filter(pl.col("gsis_id") == pid), t.filter(pl.col("gsis_id") == pid).sort("week")
    assert tp["fpoe_l1"][0] is None                       # first game: nothing before it
    assert tp["fpoe_l1"][1] == pytest.approx(gp["fpoe"][0])  # second game sees only the first


def test_current_week_is_not_final():
    # a week counts as final only once a later week's Tuesday snapshot (or a backfill) saved its finals,
    # never from the in-progress scores of its own snapshots
    from ff_model.model.backtest import final_weeks
    from ff_model.model.report import _con
    if not DB_PATH.exists():
        pytest.skip("no archive")
    con = _con()
    current = con.execute("SELECT MAX(week) FROM snapshots WHERE slot_key IS NULL OR slot_key NOT LIKE '%backfill'").fetchone()[0]
    assert current not in final_weeks(con) and current - 1 in final_weeks(con)
