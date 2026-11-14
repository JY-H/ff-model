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
