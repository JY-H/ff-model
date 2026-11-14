"""Schedule one-off pmset wakes so the Mac is awake for upcoming slots.

Needs the sudoers rule in deploy/ff-archiver.sudoers (pmset schedule is root-only).
"""
from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timedelta

from ..db import iso, utcnow
from .slots import all_slots

log = logging.getLogger("ff_model")

WAKE_LEAD = timedelta(minutes=3)
WAKE_HORIZON = timedelta(days=7)


def schedule_wakes(con, now: datetime) -> None:
    for s in all_slots(con):
        wake = s.at - WAKE_LEAD
        if not (now < wake < now + WAKE_HORIZON):
            continue
        local = wake.astimezone()  # pmset takes the Mac's local time
        stamp = local.strftime("%m/%d/%y %H:%M:%S")
        if con.execute("SELECT 1 FROM scheduled_wakes WHERE wake_local=?", (stamp,)).fetchone():
            continue
        r = subprocess.run(["sudo", "-n", "/usr/bin/pmset", "schedule", "wake", stamp],
                           capture_output=True, text=True)
        if r.returncode != 0:
            log.warning("could not schedule wake %s (sudoers rule missing?): %s",
                        stamp, r.stderr.strip()[:200])
            return
        con.execute("INSERT INTO scheduled_wakes VALUES (?,?,?)", (stamp, s.key, iso(utcnow())))
        con.commit()
        log.info("scheduled wake %s for %s", stamp, s.key)
