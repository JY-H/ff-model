"""Snapshot runner and the launchd tick that decides when to run one."""
from __future__ import annotations

import fcntl
import logging
import traceback
from datetime import datetime, timedelta

from ..config import ET, LEAGUE_ID, LOG_DIR, SEASON
from ..db import connect, iso, save_raw, utcnow
from ..sources import espn, odds
from .context import archive_context
from .parse import parse_snapshot, store_pro_games
from .slots import Slot, all_slots
from .wakes import schedule_wakes

log = logging.getLogger("ff_model")

SCHEDULE_MAX_AGE = timedelta(hours=12)


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


def _props_window(con, slot: Slot, now: datetime):
    """Games this slot is the last chance for, plus what the rest of the month needs."""
    start, end = max(slot.at, now), slot.last_kick or slot.at  # fixed slots: no props
    month_end = odds.month_end(now)
    games_later = con.execute("SELECT COUNT(*) AS n FROM pro_games WHERE season=? AND"
                              " kickoff_utc > ? AND kickoff_utc < ?",
                              (SEASON, iso(end), iso(month_end))).fetchone()["n"]
    slots_later = sum(1 for s in all_slots(con) if slot.at < s.at < month_end)
    return start, end, games_later, slots_later


def snapshot(con, week: int, slot_key: str | None = None, with_odds: bool = False,
             prev_week: bool = False, settings: bool = False, slot: Slot | None = None) -> int:
    sid = con.execute("INSERT INTO snapshots (slot_key, season, week, started_at, status)"
                      " VALUES (?,?,?,?, 'running')",
                      (slot_key, SEASON, week, iso(utcnow()))).lastrowid
    con.commit()
    log.info("snapshot %d start: week %d slot %s", sid, week, slot_key or "manual")
    kinds = ["players", "league"] + (["prev_week"] if prev_week and week > 1 else [])
    if settings or not con.execute("SELECT 1 FROM raw_responses WHERE kind='settings'"
                                   " AND status_code=200 AND league_id=? LIMIT 1", (LEAGUE_ID,)).fetchone():
        kinds.insert(0, "settings")
    errors = []
    for kind in kinds:
        try:
            for k, url, params, status, body in espn.fetch(kind, week - 1 if kind == "prev_week" else week):
                save_raw(con, sid, "espn", k, url, params, status, body)
        except Exception as e:  # keep going: a partial snapshot beats none
            log.error("fetch %s failed: %s", kind, e)
            errors.append(f"{kind}: {e}")
    if prev_week:  # Tuesday: last week's and this week's transactions (trades, claims, adds/drops)
        errors += fetch_transactions(con, [w for w in (week - 1, week) if w >= 1], sid)
    errors += archive_context(con, sid, week)  # free: nflverse + Open-Meteo
    if with_odds:
        try:
            res = odds.fetch_game_lines(con, sid)
            if res:
                save_raw(con, sid, "odds", "odds", *res)
        except Exception as e:
            log.error("game lines failed: %s", type(e).__name__)
            errors.append(f"odds: {type(e).__name__}")
    if with_odds and slot is not None:
        try:
            for res in odds.fetch_props(con, sid, *_props_window(con, slot, utcnow())):
                save_raw(con, sid, "odds", "props", *res)
        except Exception as e:
            log.error("props failed: %s", type(e).__name__)
            errors.append(f"props: {type(e).__name__}")
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


def backfill_finals(con, week: int) -> int | None:
    """Archive a finished week's locked lineups and final points (not perishable).

    Stored like a Tuesday prev_week fetch, under slot key <season>-wNN-backfill.
    """
    key = f"{SEASON}-w{week:02d}-backfill"
    if _done(con, key):
        return None
    sid = con.execute("INSERT INTO snapshots (slot_key, season, week, started_at, status)"
                      " VALUES (?,?,?,?, 'running')", (key, SEASON, week + 1, iso(utcnow()))).lastrowid
    for kind, url, params, status, body in espn.fetch("league", week):
        save_raw(con, sid, "espn", "prev_week", url, params, status, body)
    parse_snapshot(con, sid, week + 1)
    con.execute("UPDATE snapshots SET finished_at=?, status='ok' WHERE id=?", (iso(utcnow()), sid))
    con.commit()
    return sid


def fetch_transactions(con, weeks, sid: int | None = None) -> list[str]:
    """Archive ESPN's transaction log (one request per scoring period) and the league-wide
    trade feed. Returns errors."""
    errors = []
    try:
        for kind, url, params, status, body in espn.fetch("trades", 0):
            save_raw(con, sid, "espn", kind, url, params, status, body)
    except Exception as e:
        log.error("trade feed failed: %s", e)
        return [f"trades: {e}"]
    for w in weeks:
        try:
            for kind, url, params, status, body in espn.fetch("transactions", w):
                save_raw(con, sid, "espn", kind, url, params, status, body)
        except Exception as e:
            log.error("transactions week %d failed: %s", w, e)
            errors.append(f"transactions w{w}: {e}")
            break  # don't keep hammering a failing endpoint
    return errors


def _done(con, key: str) -> bool:
    return con.execute("SELECT 1 FROM snapshots WHERE slot_key=? AND status IN ('ok','partial')",
                       (key,)).fetchone() is not None


def due_slot(con, now: datetime) -> Slot | None:
    due = [s for s in all_slots(con) if s.at <= now < s.deadline and not _done(con, s.key)]
    return due[-1] if due else None


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
                     prev_week=(name == "tue"), settings=(name == "tue"), slot=slot)
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
    nxt = next((s for s in all_slots(con) if s.at > now), None)
    if nxt:
        _, _, later, slots_later = _props_window(con, nxt, now)
        games_now = con.execute("SELECT COUNT(*) AS n FROM pro_games WHERE season=? AND kickoff_utc > ?"
                                " AND kickoff_utc <= ?", (SEASON, iso(nxt.at), iso(nxt.last_kick or nxt.at))).fetchone()["n"]
        plan = odds.plan_prop_markets(odds._remaining(con), games_now, later, slots_later)
        lines.append(f"props plan if run now: {len(plan)} markets {plan} "
                     f"({games_now + later} games and {slots_later} line calls left this month)")
    return "\n".join(lines)
