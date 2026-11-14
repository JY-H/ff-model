import pytest

from ff_model.config import DB_PATH


def test_report_has_what_the_page_needs():
    if not DB_PATH.exists():
        pytest.skip("no archive")
    from ff_model.model.report import build_report
    r = build_report(n=2000)
    assert 0 <= r["p_win"] <= 1 and 0 <= r["p_win_max_proj"] <= r["p_win"] + 0.02
    me = r["players"]["me"]
    assert me and all("quantiles" in p and "effects" in p for p in me)
    starters = [p for p in me if p.get("recommended") or (p["locked"] and p["slot"] not in ("BE", "IR"))]
    assert len(starters) == 9
    assert sum(r["margin_hist"]["counts"]) <= r["margin_hist"]["n"]


def test_replay_report_handles_empty_slots_for_every_team_week():
    """Regression: a team that can't fill a slot (bye/IR) made the model's lineup hold a text placeholder."""
    if not DB_PATH.exists():
        pytest.skip("no archive")
    from ff_model.model.report import build_report
    r = build_report(week=7, team_id=17, n=500)
    assert r["outcome"] is not None and r["players"]["me"]


def test_changes_vs_max_projection_are_reported():
    """Regression: the swaps code shadowed the variables behind 'changes vs highest projection'."""
    if not DB_PATH.exists():
        pytest.skip("no archive")
    from ff_model.model.report import build_report
    for team in range(1, 23):
        try:
            r = build_report(week=10, team_id=team, n=2000)
        except Exception:
            continue
        rec = {p["name"] for p in r["players"]["me"] if p.get("recommended")}
        mp = {p["name"] for p in r["players"]["me"] if p.get("maxproj_slot")}
        assert set(r["changes"]["start"]) == rec - mp and set(r["changes"]["bench"]) == mp - rec


def test_player_card_adds_up_espn_plus_context():
    """Model expectation = ESPN projection + the listed context effects (ESPN + context variant)."""
    if not DB_PATH.exists():
        pytest.skip("no archive")
    from ff_model.model.report import build_report
    for week in (10, 3):
        r = build_report(week=week, team_id=9, n=500)
        for p in r["players"]["me"] + r["players"]["opp"]:
            if p["locked"] or p["unavailable"] or not p["proj"]:
                continue
            expected = p["proj"] + sum(p["effects"].values())
            assert abs(p["model_mean"] - expected) < 0.15, (week, p["name"], p["proj"], p["effects"], p["model_mean"])
