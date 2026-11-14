"""Polite fetchers for ESPN's unofficial fantasy API.

Every response is returned as (url, params, status, body) so the caller can
archive it verbatim. Cookies are attached here and never logged.
"""
from __future__ import annotations

import json
import logging
import time

import requests

from ..config import ESPN_LEAGUE_URL, ESPN_SEASON_URL, espn_cookies

log = logging.getLogger(__name__)

REQUEST_GAP_S = 1.5
TIMEOUT_S = 30
PAGE_SIZE = 500

_session: requests.Session | None = None
_last_request = 0.0


def _get(url: str, params: dict, filt: dict | None = None):
    global _session, _last_request
    if _session is None:
        _session = requests.Session()
        _session.headers["User-Agent"] = "ff-model-archiver (personal use)"
        _session.cookies.update(espn_cookies())
    headers = {"x-fantasy-filter": json.dumps(filt)} if filt else {}
    for attempt in range(4):
        wait = REQUEST_GAP_S - (time.monotonic() - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.monotonic()
        try:
            r = _session.get(url, params=params, headers=headers, timeout=TIMEOUT_S)
        except requests.RequestException as e:
            log.warning("ESPN request failed (%s), attempt %d", type(e).__name__, attempt + 1)
        else:
            if r.status_code == 200:
                return r.status_code, r.json()
            if r.status_code in (401, 403, 404, 429):  # 429: rate-limited; stop rather than retry
                raise RuntimeError(f"ESPN returned HTTP {r.status_code}" + (" (cookies expired?)" if r.status_code != 429 else ""))
            log.warning("ESPN HTTP %d, attempt %d", r.status_code, attempt + 1)
        time.sleep(5 * 2**attempt)
    raise RuntimeError("ESPN request failed after retries")


def fetch(kind: str, week: int):
    """Yield (kind, url, params_for_archive, status, body) for one logical fetch."""
    if kind == "settings":
        params = {"view": ["mSettings", "mStatus"]}
        yield (kind, ESPN_LEAGUE_URL, params, *_get(ESPN_LEAGUE_URL, params))
    elif kind == "pro_schedule":
        params = {"view": "proTeamSchedules_wl"}
        yield (kind, ESPN_SEASON_URL, params, *_get(ESPN_SEASON_URL, params))
    elif kind in ("league", "prev_week"):
        # Rosters with lineup slots + per-player stats, and this period's matchups.
        params = {"view": ["mTeam", "mRoster", "mMatchup", "mMatchupScore", "mStatus"],
                  "scoringPeriodId": week}
        filt = {"schedule": {"filterMatchupPeriodIds": {"value": [week]}}}
        yield (kind, ESPN_LEAGUE_URL, {**params, "x-fantasy-filter": filt},
               *_get(ESPN_LEAGUE_URL, params, filt))
    elif kind == "players":
        # The whole fantasy player pool (rostered, FA, waivers), paged.
        offset = 0
        while True:
            params = {"view": "kona_player_info", "scoringPeriodId": week}
            filt = {"players": {
                "filterStatus": {"value": ["FREEAGENT", "WAIVERS", "ONTEAM"]},
                "limit": PAGE_SIZE, "offset": offset,
                "sortPercOwned": {"sortPriority": 1, "sortAsc": False},
            }}
            status, body = _get(ESPN_LEAGUE_URL, params, filt)
            yield (kind, ESPN_LEAGUE_URL, {**params, "x-fantasy-filter": filt}, status, body)
            n = len(body.get("players", []))
            if n < PAGE_SIZE:
                break
            offset += PAGE_SIZE
    elif kind == "transactions":
        # Every roster transaction processed in one scoring period: trades, waiver claims, adds/drops.
        params = {"view": "mTransactions2", "scoringPeriodId": week}
        yield (kind, ESPN_LEAGUE_URL, params, *_get(ESPN_LEAGUE_URL, params))
    else:
        raise ValueError(kind)
