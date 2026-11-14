from datetime import timedelta

from ff_model import config
from ff_model.archive.slots import week_slots
from ff_model.sources.odds import plan_prop_markets

from test_archive_slots import KICKS


def test_full_markets_when_budget_is_ample():
    assert plan_prop_markets(493, 13, 32, 25) == config.PROP_MARKETS


def test_drops_low_priority_markets_first():
    # 400 credits; 80 games left; 50 line calls (100 credits); floor 40 -> 260 / 80 = 3 markets
    assert plan_prop_markets(400, 16, 64, 50) == config.PROP_MARKETS[:3]


def test_keeps_td_odds_for_current_games_when_month_is_unaffordable():
    assert plan_prop_markets(100, 10, 200, 20) == ["player_anytime_td"]
    assert plan_prop_markets(45, 10, 200, 20) == []


def test_each_game_falls_in_exactly_one_slot_window():
    slots = week_slots(10, KICKS)
    for k in KICKS:
        owners = [s.key for s in slots if s.last_kick and s.at < k <= s.last_kick]
        assert len(owners) == 1, (k, owners)
        s = next(s for s in slots if s.key == owners[0])
        assert k - s.at <= timedelta(minutes=90) or s.key.endswith("sun_1145")
