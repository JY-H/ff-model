import pytest

from ff_model.config import DB_PATH

COUNTS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "D/ST": 1, "K": 1, "FLEX": 1}


def test_best_lineup_fills_fixed_slots_then_flex():
    from ff_model.model.trades import _best_lineup
    ps = [{"id": i, "pos": pos, "proj": proj} for i, (pos, proj) in enumerate(
        [("QB", 20), ("QB", 25), ("RB", 15), ("RB", 12), ("RB", 11), ("WR", 14), ("WR", 13), ("WR", 9),
         ("TE", 8), ("TE", 10), ("D/ST", 6), ("K", 7)])]
    ids = {p["id"] for p in _best_lineup(ps, COUNTS)}
    assert len(ids) == 9
    assert 1 in ids and 0 not in ids          # better QB
    assert 9 in ids and 8 not in ids          # better TE
    assert 4 in ids and 7 not in ids          # FLEX = best leftover RB/WR/TE (11 > 9; TE 8 < 11)


@pytest.mark.skipif(not DB_PATH.exists(), reason="needs the local archive")
def test_trade_lineup_diff_adds_up():
    from ff_model.model.trades import analyze
    a = analyze()
    for t in a["trades"]:
        assert t["net_lineup"] == pytest.approx(sum(w["with_trade"] - w["undone"] for w in t["weekly"]))
        assert all(w["week"] >= t["week"] for w in t["weekly"])
