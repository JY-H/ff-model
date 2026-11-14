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
