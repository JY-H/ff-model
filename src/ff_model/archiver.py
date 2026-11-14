"""Snapshot runner, launchd tick, and pmset wake scheduling."""
from __future__ import annotations

import fcntl
import logging
import subprocess
import traceback
from datetime import datetime, timedelta

from . import espn, odds
from .config import ET, LOG_DIR, SEASON
from .db import connect, iso, save_raw, utcnow
from .parse import parse_snapshot, store_pro_games
from .slots import Slot, all_slots

log = logging.getLogger("ff_model")

SCHEDULE_MAX_AGE = timedelta(hours=12)
WAKE_LEAD = timedelta(minutes=3)
WAKE_HORIZON = timedelta(days=7)


def refresh_pro_schedule(con, force=False) -> None:
    row = con.execute("SELECT MAX(refreshed_at) AS t FROM pro_games WHERE season=?",
                      (SEASON,)).fetchone()
    if not force and row["t"] and datetime.fromisoformat(row["t"]) > utcnow() - SCHEDULE_MAX_AGE:
        return
    for kind, url, params, status, body in espn.fetch("pro_schedule", 0):
        save_raw(con, None, "espn", kind, url, params, status, body)
        n = store_pro_games(con, SEASON, body)
        log.info("refreshed pro schedule: %d games", n)


def current_week(con) -> int:
    """The earliest week that still has a game in the future."""
    row = con.execute("SELECT week FROM pro_games WHERE season=? GROUP BY week"
                      " HAVING MAX(kickoff_utc) > ? ORDER BY week LIMIT 1",
                      (SEASON, iso(utcnow()))).fetchone()
    if row is None:
        raise RuntimeError("no remaining games in schedule")
    return row["week"]


def snapshot(con, week: int, slot_key: str | None = None, with_odds: bool = False,
             prev_week: bool = False, settings: bool = False) -> int:
    sid = con.execute("INSERT INTO snapshots (slot_key, season, week, started_at, status)"
                      " VALUES (?,?,?,?, 'running')",
                      (slot_key, SEASON, week, iso(utcnow()))).lastrowid
    con.commit()
    log.info("snapshot %d start: week %d slot %s", sid, week, slot_key or "manual")
    kinds = ["players", "league"] + (["prev_week"] if prev_week and week > 1 else [])
    if settings or not con.execute("SELECT 1 FROM raw_responses WHERE kind='settings'"
                                   " AND status_code=200 LIMIT 1").fetchone():
        kinds.insert(0, "settings")
    errors = []
    for kind in kinds:
        try:
            for k, url, params, status, body in espn.fetch(kind, week - 1 if kind == "prev_week" else week):
                save_raw(con, sid, "espn", k, url, params, status, body)
        except Exception as e:  # keep going: a partial snapshot beats none
            log.error("fetch %s failed: %s", kind, e)
            errors.append(f"{kind}: {e}")
    if with_odds:
        try:
            res = odds.fetch(con, sid)
            if res:
                save_raw(con, sid, "odds", "odds", *res)
        except Exception as e:
            log.error("odds failed: %s", type(e).__name__)
            errors.append(f"odds: {type(e).__name__}")
    try:
        parse_snapshot(con, sid, week)
    except Exception:
        log.error("parse failed:\n%s", traceback.format_exc())
        errors.append("parse")
    ok_espn = con.execute("SELECT COUNT(*) AS n FROM raw_responses WHERE snapshot_id=? AND"
                          " source='espn' AND status_code=200", (sid,)).fetchone()["n"]
    status = "ok" if not errors else ("partial" if ok_espn else "failed")
    con.execute("UPDATE snapshots SET finished_at=?, status=?, error=? WHERE id=?",
                (iso(utcnow()), status, "; ".join(errors) or None, sid))
    con.commit()
    log.info("snapshot %d %s", sid, status)
    return sid


def _done(con, key: str) -> bool:
    return con.execute("SELECT 1 FROM snapshots WHERE slot_key=? AND status IN ('ok','partial')",
                       (key,)).fetchone() is not None


def due_slot(con, now: datetime) -> Slot | None:
    due = [s for s in all_slots(con) if s.at <= now < s.deadline and not _done(con, s.key)]
    return due[-1] if due else None


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


def tick() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    with open(LOG_DIR / "tick.lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log.info("another tick is running")
            return
        con = connect()
        now = utcnow().astimezone(ET)
        try:
            refresh_pro_schedule(con)
        except Exception as e:
            log.error("schedule refresh failed: %s", e)
        slot = due_slot(con, now)
        if slot:
            name = slot.key.rsplit("-", 1)[1]
            snapshot(con, slot.week, slot.key, with_odds=True,
                     prev_week=(name == "tue"), settings=(name == "tue"))
        try:
            schedule_wakes(con, now)
        except Exception as e:
            log.error("wake scheduling failed: %s", e)


def status_report(con) -> str:
    lines = []
    now = utcnow().astimezone(ET)
    for s in all_slots(con):
        if now - timedelta(days=3) < s.at < now + timedelta(days=8):
            done = "done" if _done(con, s.key) else ("MISSED" if s.deadline <= now else "")
            lines.append(f"{s.at:%a %m-%d %H:%M} ET  {s.key:<24} {done:<6} {s.reason}")
    for r in con.execute("SELECT id, slot_key, started_at, status, error FROM snapshots"
                         " ORDER BY id DESC LIMIT 8"):
        lines.append(f"snapshot {r['id']}: {r['slot_key'] or 'manual'} {r['started_at']} {r['status']}"
                     + (f" ({r['error']})" if r["error"] else ""))
    c = odds.credits_status(con)
    lines.append(f"odds credits remaining: {c['credits_remaining'] if c else 'unknown (no calls yet)'}")
    return "\n".join(lines)
