"""Backfill ESPN's weekly projections for past seasons (2019-2025).

ESPN's public league-defaults endpoint (no cookies needed) still serves
statSourceId=1 weekly projections, with full projected stat lines, back to
2019; 2018 and earlier return nothing. Raw responses are kept as gzipped JSON
under data/espn_history/<season>/w<week>-p<page>.json.gz; parse() builds a
tidy Parquet table from them.

Caveats:
- It is unverified whether these are the projections as of lineup lock or a
  later revision. Compare against the live archive (2026 W10+) before relying
  on them.
- injuryStatus etc. in these responses are *current*, not as of that week.
  Never use them historically.
- appliedTotal is under ESPN default PPR scoring, not this league's; rescore
  from the projected stat lines.
"""
from __future__ import annotations

import gzip
import json
import logging
import time

import polars as pl
import requests
from espn_api.football.constant import PLAYER_STATS_MAP

from ..config import DATA_DIR

log = logging.getLogger(__name__)

HIST_DIR = DATA_DIR / "espn_history"
FIRST_SEASON, LAST_SEASON = 2019, 2025
URL = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}/segments/0/leaguedefaults/3"
PAGE = 1000
GAP_S = 4.0              # ~250 requests total -> ~17 min; be gentle
MAX_PER_RUN = 300        # hard cap per invocation
STOP_STATUSES = {401, 403, 429}  # rate-limited or blocked: stop, don't retry


class Throttled(RuntimeError):
    pass


def weeks_in(season: int) -> range:
    return range(1, 18 if season <= 2020 else 19)


def _path(season, week, page):
    return HIST_DIR / str(season) / f"w{week:02d}-p{page}.json.gz"


def download(seasons=None) -> None:
    s = requests.Session()
    s.headers["User-Agent"] = "ff-model-archiver (personal use)"
    sent = 0
    for season in seasons or range(FIRST_SEASON, LAST_SEASON + 1):
        for week in weeks_in(season):
            page = 0
            while True:
                path = _path(season, week, page)
                if path.exists():
                    n = len(json.loads(gzip.decompress(path.read_bytes())).get("players", []))
                else:
                    filt = {"players": {
                        "limit": PAGE, "offset": page * PAGE,
                        "sortPercOwned": {"sortPriority": 1, "sortAsc": False},
                        "filterStatsForTopScoringPeriodIds": {"value": 1, "additionalValue": [f"11{season}{week}"]},
                    }}
                    for attempt in range(3):
                        if sent >= MAX_PER_RUN:
                            raise Throttled(f"hit MAX_PER_RUN={MAX_PER_RUN}; rerun later to resume")
                        time.sleep(GAP_S * 4**attempt)  # 4s, then 16s, 64s on transient errors
                        sent += 1
                        r = s.get(URL.format(season=season), timeout=60,
                                  params={"view": "kona_player_info", "scoringPeriodId": week},
                                  headers={"x-fantasy-filter": json.dumps(filt)})
                        if r.status_code == 200:
                            break
                        log.warning("%d w%d p%d: HTTP %d", season, week, page, r.status_code)
                        if r.status_code in STOP_STATUSES:
                            raise Throttled(f"ESPN returned HTTP {r.status_code}; stopping. Wait before resuming.")
                    r.raise_for_status()
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(gzip.compress(r.content))
                    n = len(r.json().get("players", []))
                    log.info("%d w%d p%d: %d players", season, week, page, n)
                if n < PAGE:
                    break
                page += 1


def parse(seasons=None, write: bool = True) -> pl.DataFrame:
    """Tidy weekly projections: one row per player-week, stat lines as st_<espn stat id>."""
    rows = []
    for path in sorted(HIST_DIR.glob("*/w*-p*.json.gz")):
        if seasons and int(path.parent.name) not in seasons:
            continue
        season, week = int(path.parent.name), int(path.name[1:3])
        for w in json.loads(gzip.decompress(path.read_bytes())).get("players", []):
            p = w["player"]
            for st in p.get("stats", []):
                if (st.get("statSourceId"), st.get("statSplitTypeId"), st.get("scoringPeriodId"),
                        st.get("seasonId")) != (1, 1, week, season):
                    continue
                row = {"season": season, "week": week, "espn_id": p["id"], "name": p.get("fullName"),
                       "position_id": p.get("defaultPositionId"), "pro_team_id": st.get("proTeamId"),
                       "proj_ppr_default": st.get("appliedTotal")}
                for k, v in (st.get("stats") or {}).items():
                    row[f"st_{k}"] = v
                rows.append(row)
    df = pl.DataFrame(rows, infer_schema_length=None).unique(["season", "week", "espn_id"])
    if write:
        df.write_parquet(HIST_DIR / "projections.parquet")
    return df


def stat_name(col: str) -> str:
    """st_53 -> receivingReceptions, for display."""
    return PLAYER_STATS_MAP.get(int(col.removeprefix("st_")), col)
