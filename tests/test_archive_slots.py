from datetime import datetime, timedelta

import pytest

from ff_model.archive import runner
from ff_model.config import ET
from ff_model.archive.slots import week_slots

# Week 10, 2026: TNF, London 9:30, 1pm, 4:05/4:25, SNF 8:20, MNF 8:15
KICKS = [datetime(2026, 11, 12, 20, 15, tzinfo=ET), datetime(2026, 11, 15, 9, 30, tzinfo=ET),
         datetime(2026, 11, 15, 13, 0, tzinfo=ET), datetime(2026, 11, 15, 16, 5, tzinfo=ET),
         datetime(2026, 11, 15, 16, 25, tzinfo=ET), datetime(2026, 11, 15, 20, 20, tzinfo=ET),
         datetime(2026, 11, 16, 20, 15, tzinfo=ET)]


def test_week_slots():
    got = {s.key.split("-")[-1]: s.at.strftime("%a %H:%M") for s in week_slots(10, KICKS)}
    assert got == {"tue": "Tue 12:00", "wed": "Wed 12:00", "thu_1915": "Thu 19:15",
                   "sat": "Sat 22:00", "sun_0800": "Sun 08:00", "sun_1145": "Sun 11:45",
                   "sun_1555": "Sun 15:55", "sun_1920": "Sun 19:20", "mon_1915": "Mon 19:15"}


def test_every_slot_precedes_the_lock_it_serves():
    for s in week_slots(10, KICKS):
        assert s.at < s.deadline
        assert all(not (s.at < k <= s.at + timedelta(minutes=1)) for k in KICKS)


def test_week_without_thursday_game():
    kicks = [datetime(2027, 1, 9, 16, 30, tzinfo=ET), datetime(2027, 1, 10, 13, 0, tzinfo=ET)]
    got = {s.key.split("-")[-1]: s.at.strftime("%m-%d %a %H:%M") for s in week_slots(18, kicks)}
    assert got["tue"] == "01-05 Tue 12:00"
    assert got["sat_1530"] == "01-09 Sat 15:30"
    assert "sat" not in got or got["sat"] == "01-09 Sat 22:00"


@pytest.mark.parametrize("now, expect", [
    (datetime(2026, 11, 15, 8, 5, tzinfo=ET), "sun_0800"),
    (datetime(2026, 11, 15, 9, 31, tzinfo=ET), None),       # intl game already locked
    (datetime(2026, 11, 15, 12, 59, tzinfo=ET), "sun_1145"),  # late wake still in time
    (datetime(2026, 11, 15, 13, 1, tzinfo=ET), None),
    (datetime(2026, 11, 15, 2, 0, tzinfo=ET), "sat"),        # catch up missed Sat run
])
def test_due_slot(monkeypatch, now, expect):
    monkeypatch.setattr(runner, "all_slots", lambda con: week_slots(10, KICKS))
    monkeypatch.setattr(runner, "_done", lambda con, key: False)
    s = runner.due_slot(None, now)
    assert (s.key.split("-")[-1] if s else None) == expect
