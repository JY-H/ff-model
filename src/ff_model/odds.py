"""Spreads and totals from The Odds API, with a hard credit budget.

Free tier = 500 credits/month; each call here costs ODDS_CREDITS_PER_CALL.
The API reports the remaining balance in response headers; we record it in
odds_ledger and refuse to call once it drops below ODDS_REMAINING_FLOOR.
"""
from __future__ import annotations

import logging
from datetime import timedelta

import requests

from . import config
from .db import iso, utcnow

log = logging.getLogger(__name__)


def budget_check(con) -> str | None:
    """Return a reason to skip, or None if a call is allowed."""
    if not config.odds_api_key():
        return "no ODDS_API_KEY"
    row = con.execute(
        "SELECT fetched_at, credits_remaining FROM odds_ledger"
        " WHERE credits_remaining IS NOT NULL ORDER BY fetched_at DESC LIMIT 1").fetchone()
    if row is None:
        return None
    month_now = utcnow().strftime("%Y-%m")
    # Balance resets monthly; a reading from an earlier month is stale.
    if row["credits_remaining"] - config.ODDS_CREDITS_PER_CALL < config.ODDS_REMAINING_FLOOR \
            and row["fetched_at"][:7] == month_now:
        return f"only {row['credits_remaining']} credits left (floor {config.ODDS_REMAINING_FLOOR})"
    last = con.execute("SELECT MAX(fetched_at) AS t FROM odds_ledger").fetchone()["t"]
    if last and last > iso(utcnow() - timedelta(minutes=config.ODDS_MIN_GAP_MINUTES)):
        return f"odds fetched within the last {config.ODDS_MIN_GAP_MINUTES} min"
    return None


def fetch(con, snapshot_id: int):
    """Return (url, params_for_archive, status, body), or None if skipped."""
    reason = budget_check(con)
    if reason:
        log.info("skipping odds: %s", reason)
        return None
    params = {"regions": config.ODDS_REGIONS, "markets": config.ODDS_MARKETS,
              "oddsFormat": "american", "dateFormat": "iso"}
    try:
        r = requests.get(config.ODDS_URL, params={**params, "apiKey": config.odds_api_key()},
                         timeout=30)
    except requests.RequestException as e:
        # Don't log str(e): it can contain the request URL, which has the key.
        log.warning("odds request failed: %s", type(e).__name__)
        return None

    def h(name):
        v = r.headers.get(name)
        return int(float(v)) if v is not None else None

    con.execute("INSERT INTO odds_ledger VALUES (?,?,?,?,?,?)",
                (iso(utcnow()), snapshot_id, r.status_code, h("x-requests-last"),
                 h("x-requests-used"), h("x-requests-remaining")))
    con.commit()
    log.info("odds HTTP %d, credits remaining %s", r.status_code, h("x-requests-remaining"))
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else None
    return config.ODDS_URL, params, r.status_code, body


def credits_status(con) -> dict | None:
    row = con.execute("SELECT * FROM odds_ledger ORDER BY fetched_at DESC LIMIT 1").fetchone()
    return dict(row) if row else None
