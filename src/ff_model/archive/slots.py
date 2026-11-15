"""When to snapshot.

Fixed slots each week (ET):
  tue      Tue 12:00          start of week; also archives last week's final scores
  wed      Wed 12:00          after this league's Wednesday waiver run (no Tue waivers)
  sat      Sat 22:00
Kickoff-relative slots, one per distinct kickoff time:
  Sunday 1pm window          Sun 11:45 (after inactives)
  Sunday 4pm window          Sun 15:55 (its props are pulled at 11:45, with the 1pm games')
  Sunday before noon (intl)  kickoff - 90 min
  anything else (TNF, SNF, MNF, Sat/Fri games)   kickoff - 60 min
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta

from ..config import ET, SEASON


@dataclass(frozen=True)
class Slot:
    key: str
    week: int
    at: datetime        # tz-aware ET
    deadline: datetime  # don't run after this (the lock it serves, or next slot)
    reason: str
    last_kick: datetime | None = None  # latest kickoff this slot pulls props for


def _at(day: datetime, hh: int, mm: int) -> datetime:
    return datetime.combine(day.date(), time(hh, mm), ET)


def week_slots(week: int, kickoffs: list[datetime]) -> list[Slot]:
    if not kickoffs:
        return []
    kicks = sorted({k.astimezone(ET) for k in kickoffs})
    first, last = kicks[0], kicks[-1]
    tue = first - timedelta(days=(first.weekday() - 1) % 7)
    raw: dict[datetime, tuple[str, str, datetime]] = {}
    covers: dict[datetime, datetime] = {}

    def add(at, name, reason, deadline):
        if at < deadline and at not in raw:
            raw[at] = (name, reason, deadline)

    for k in kicks:
        if k.weekday() == 6 and 12 <= k.hour < 15:
            at, why = _at(k, 11, 45), "before Sunday early games"
        elif k.weekday() == 6 and 15 <= k.hour < 18:
            at, why = _at(k, 15, 55), "before Sunday late games"
        elif k.weekday() == 6 and k.hour < 12:
            at, why = k - timedelta(minutes=90), "90 min before international game"
        else:
            at, why = k - timedelta(minutes=60), f"60 min before {k:%a %H:%M} kickoff"
        add(at, f"{at:%a_%H%M}".lower(), why, k)
        covers[at] = max(covers.get(at, k), k)

    # late Sunday games' props come with the early games' at 11:45, so start/sit calls made before
    # 1 PM can use them; 15:55 still refreshes ESPN, injuries, game lines and weather
    sun = {(at.hour, at.minute): at for at in covers if at.weekday() == 6}
    if (11, 45) in sun and (15, 55) in sun:
        covers[sun[11, 45]] = covers.pop(sun[15, 55])

    add(_at(tue, 12, 0), "tue", "start of week + last week's finals", last)
    add(_at(tue + timedelta(days=1), 12, 0), "wed", "after Wednesday waivers", last)
    add(_at(tue + timedelta(days=4), 22, 0), "sat", "Saturday night", last)

    out = []
    times = sorted(raw)
    for i, at in enumerate(times):
        name, why, lock = raw[at]
        nxt = times[i + 1] if i + 1 < len(times) else lock
        out.append(Slot(f"{SEASON}-w{week:02d}-{name}", week, at,
                        min(lock, max(nxt, at + timedelta(minutes=20))), why, covers.get(at)))
    return out


def all_slots(con) -> list[Slot]:
    weeks: dict[int, list[datetime]] = {}
    for r in con.execute("SELECT week, kickoff_utc FROM pro_games WHERE season=?", (SEASON,)):
        weeks.setdefault(r["week"], []).append(datetime.fromisoformat(r["kickoff_utc"]))
    return sorted((s for w, ks in weeks.items() for s in week_slots(w, ks)), key=lambda s: s.at)
