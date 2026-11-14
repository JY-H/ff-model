"""Game lines and player props from The Odds API, under a monthly credit budget.

Every paid call is recorded in odds_ledger with the balance the API reports in
its response headers. Game lines are fetched at every snapshot slot; props are
fetched once per game at the slot just before it locks, with the number of
markets chosen by plan_prop_markets so the month's remaining games still fit.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import requests

from .. import config
from ..db import iso, utcnow

log = logging.getLogger(__name__)


def _remaining(con) -> int | None:
    """Last reported balance, or None if unknown or from a previous month."""
    row = con.execute("SELECT fetched_at, credits_remaining FROM odds_ledger WHERE"
                      " credits_remaining IS NOT NULL ORDER BY fetched_at DESC LIMIT 1").fetchone()
    if row is None or row["fetched_at"][:7] != utcnow().strftime("%Y-%m"):
        return None
    return row["credits_remaining"]


def _call(con, snapshot_id, kind, path, params):
    """GET a paid endpoint; returns (url, params_for_archive, status, body) or None."""
    url = f"{config.ODDS_BASE}{path}"
    try:
        r = requests.get(url, params={**params, "apiKey": config.odds_api_key()}, timeout=30)
    except requests.RequestException as e:
        # Don't log str(e): it can contain the request URL, which has the key.
        log.warning("odds %s request failed: %s", kind, type(e).__name__)
        return None

    def h(name):
        v = r.headers.get(name)
        return int(float(v)) if v is not None else None

    con.execute("INSERT INTO odds_ledger (fetched_at, snapshot_id, status_code, credits_last,"
                " credits_used, credits_remaining, kind) VALUES (?,?,?,?,?,?,?)",
                (iso(utcnow()), snapshot_id, r.status_code, h("x-requests-last"),
                 h("x-requests-used"), h("x-requests-remaining"), kind))
    con.commit()
    log.info("odds %s HTTP %d, cost %s, remaining %s", kind, r.status_code,
             h("x-requests-last"), h("x-requests-remaining"))
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else None
    return url, params, r.status_code, body


def fetch_game_lines(con, snapshot_id: int):
    if not config.odds_api_key():
        log.info("skipping game lines: no ODDS_API_KEY")
        return None
    rem = _remaining(con)
    if rem is not None and rem - config.ODDS_CREDITS_PER_CALL < config.ODDS_REMAINING_FLOOR:
        log.info("skipping game lines: only %d credits left", rem)
        return None
    last = con.execute("SELECT MAX(fetched_at) AS t FROM odds_ledger WHERE kind='odds'").fetchone()["t"]
    if last and last > iso(utcnow() - timedelta(minutes=config.ODDS_MIN_GAP_MINUTES)):
        log.info("skipping game lines: fetched within %d min", config.ODDS_MIN_GAP_MINUTES)
        return None
    return _call(con, snapshot_id, "odds", "/odds",
                 {"regions": config.ODDS_REGIONS, "markets": config.ODDS_MARKETS,
                  "oddsFormat": "american", "dateFormat": "iso"})


def fetch_events():
    """Upcoming events with ids and commence times. Free (costs no credits)."""
    r = requests.get(f"{config.ODDS_BASE}/events",
                     params={"apiKey": config.odds_api_key(), "dateFormat": "iso"}, timeout=30)
    r.raise_for_status()
    return r.json()


def month_end(now: datetime) -> datetime:
    first = now.astimezone(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return (first + timedelta(days=32)).replace(day=1)


def plan_prop_markets(remaining: int | None, n_games_now: int, n_games_later: int,
                      n_line_calls_later: int) -> list[str]:
    """Most prop markets we can afford for every game left this month.

    Reserves credits for game lines at every remaining slot first, then picks
    the largest prefix of PROP_MARKETS that covers props for all remaining games.
    """
    if remaining is None:  # unknown balance (new month): assume a fresh 500
        remaining = 500
    spendable = remaining - config.ODDS_REMAINING_FLOOR - n_line_calls_later * config.ODDS_CREDITS_PER_CALL
    games = n_games_now + n_games_later
    for k in range(len(config.PROP_MARKETS), 0, -1):
        if games * k <= spendable:
            return config.PROP_MARKETS[:k]
    # Can't cover the month: still take TD odds for these games if they fit.
    return config.PROP_MARKETS[:1] if n_games_now <= spendable else []


def fetch_props(con, snapshot_id: int, window_start: datetime, window_end: datetime,
                n_games_later: int, n_line_calls_later: int):
    """Props for games kicking off in (window_start, window_end] not already archived.

    Yields (url, params_for_archive, status, body) per game.
    """
    if not config.odds_api_key():
        return
    try:
        events = fetch_events()
    except requests.RequestException as e:
        log.warning("odds events failed: %s", type(e).__name__)
        return
    todo = []
    for ev in events:
        kick = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
        if window_start < kick <= window_end and not con.execute(
                "SELECT 1 FROM raw_responses WHERE kind='props' AND status_code=200"
                " AND json_extract(params_json, '$.event_id')=?", (ev["id"],)).fetchone():
            todo.append(ev)
    if not todo:
        return
    markets = plan_prop_markets(_remaining(con), len(todo), n_games_later, n_line_calls_later)
    log.info("props for %d games, markets %s", len(todo), markets)
    for ev in todo:
        if not markets:
            log.info("skipping props: budget exhausted")
            return
        rem = _remaining(con)
        if rem is not None and rem - len(markets) < config.ODDS_REMAINING_FLOOR:
            log.info("skipping props: only %d credits left", rem)
            return
        res = _call(con, snapshot_id, "props", f"/events/{ev['id']}/odds",
                    {"regions": config.ODDS_REGIONS, "markets": ",".join(markets),
                     "oddsFormat": "american", "dateFormat": "iso"})
        if res:
            url, params, status, body = res
            yield url, {**params, "event_id": ev["id"]}, status, body


def credits_status(con) -> dict | None:
    row = con.execute("SELECT * FROM odds_ledger ORDER BY fetched_at DESC LIMIT 1").fetchone()
    return dict(row) if row else None
