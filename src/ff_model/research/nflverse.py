"""Download nflverse history to data/nflverse/<dataset>/<season>.parquet.

Completed seasons are fetched once; the current season is re-fetched on
request (it grows every week). Nothing here is perishable, so this runs by
hand, not from launchd.

Leakage: the current season's files include games already played this week.
Anything that builds features for week W must filter to weeks < W itself.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import nflreadpy as nfl
import polars as pl

from ..config import DATA_DIR, SEASON

log = logging.getLogger(__name__)

NFLVERSE_DIR = DATA_DIR / "nflverse"
FIRST_SEASON = 2012  # snap counts start here

# name -> loader taking one season
SEASONAL = {
    "player_stats": lambda s: nfl.load_player_stats(s, summary_level="week"),
    "team_stats": lambda s: nfl.load_team_stats(s, summary_level="week"),
    "snap_counts": nfl.load_snap_counts,
    "schedules": nfl.load_schedules,
    "injuries": nfl.load_injuries,
    "ff_opportunity": lambda s: nfl.load_ff_opportunity(s, stat_type="weekly"),
    "rosters_weekly": nfl.load_rosters_weekly,  # each player's team per week (known pre-game)
}
# name -> loader for tables that aren't split by season
STATIC = {
    "ff_playerids": nfl.load_ff_playerids,  # crosswalk incl. espn_id <-> gsis_id
}


def _write(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.with_columns(pl.lit(datetime.now(timezone.utc)).alias("_pulled_at"))
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


def pull(seasons: list[int] | None = None, refresh_current: bool = True, force: bool = False,
         datasets: list[str] | None = None) -> None:
    seasons = seasons or list(range(FIRST_SEASON, SEASON + 1))
    for name, loader in SEASONAL.items():
        if datasets is not None and name not in datasets:
            continue
        for s in seasons:
            path = NFLVERSE_DIR / name / f"{s}.parquet"
            stale = force or not path.exists() or (s == SEASON and refresh_current)
            if not stale:
                continue
            try:
                df = loader(s)
            except Exception as e:  # some datasets lag or skip seasons
                log.warning("%s %d: %s", name, s, e)
                continue
            _write(df, path)
            log.info("%s %d: %d rows", name, s, df.height)
    for name, loader in STATIC.items():
        if datasets is not None and name not in datasets:
            continue
        df = loader()
        _write(df, NFLVERSE_DIR / f"{name}.parquet")
        log.info("%s: %d rows", name, df.height)


def load(name: str, seasons: list[int] | None = None) -> pl.DataFrame:
    """Read a dataset back, tolerating column drift across seasons."""
    if name in STATIC:
        return pl.read_parquet(NFLVERSE_DIR / f"{name}.parquet")
    files = sorted((NFLVERSE_DIR / name).glob("*.parquet"))
    if seasons is not None:
        files = [f for f in files if int(f.stem) in seasons]
    return pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")


def inventory() -> pl.DataFrame:
    rows = []
    for name in SEASONAL:
        for f in sorted((NFLVERSE_DIR / name).glob("*.parquet")):
            meta = pl.read_parquet(f, columns=["_pulled_at"]).head(1)
            rows.append({"dataset": name, "season": int(f.stem),
                         "rows": pl.scan_parquet(f).select(pl.len()).collect().item(),
                         "pulled_at": meta["_pulled_at"][0] if meta.height else None})
    return pl.DataFrame(rows)
