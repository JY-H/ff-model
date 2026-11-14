"""SQLite storage. Raw responses are the source of truth; other tables are
derived from them by parse.py and can be rebuilt.

League-specific data carries league_id (ESPN league): raw responses from league endpoints,
player_snapshots (league-scored, with ownership), lineups and matchups. Shared data (odds,
props, weather, injury reports, the NFL schedule) has league_id NULL and serves every league.
Readers filter on config.LEAGUE_ID."""
from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
from datetime import datetime, timezone

from .config import DATA_DIR, DB_PATH, LEAGUE_ID

# ESPN responses from league endpoints (settings, rosters, the league-scored player pool, transactions)
LEAGUE_KINDS = ("settings", "league", "prev_week", "players", "transactions", "trades")
LEAGUE_TABLES = ("player_snapshots", "lineups", "matchups")

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY,
    slot_key TEXT,              -- e.g. 2026-w10-sun_1145; NULL for manual runs
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    started_at TEXT NOT NULL,   -- UTC ISO
    finished_at TEXT,
    status TEXT NOT NULL,       -- running | ok | partial | failed
    error TEXT
);
CREATE INDEX IF NOT EXISTS snapshots_slot ON snapshots(slot_key);

CREATE TABLE IF NOT EXISTS raw_responses (
    id INTEGER PRIMARY KEY,
    snapshot_id INTEGER REFERENCES snapshots(id),  -- NULL for maintenance fetches
    source TEXT NOT NULL,       -- espn | odds
    kind TEXT NOT NULL,         -- players | league | settings | pro_schedule | odds | prev_week
    url TEXT NOT NULL,
    params_json TEXT,           -- never contains credentials
    fetched_at TEXT NOT NULL,
    status_code INTEGER,
    sha256 TEXT,
    body_gz BLOB,
    league_id INTEGER           -- ESPN league for league endpoints; NULL for shared data
);
CREATE INDEX IF NOT EXISTS raw_snapshot ON raw_responses(snapshot_id, kind);

CREATE TABLE IF NOT EXISTS player_snapshots (
    league_id INTEGER NOT NULL,
    snapshot_id INTEGER NOT NULL,
    player_id INTEGER NOT NULL,
    full_name TEXT,
    position_id INTEGER,
    pro_team_id INTEGER,
    status TEXT,                -- ONTEAM | FREEAGENT | WAIVERS
    on_team_id INTEGER,
    injury_status TEXT,
    injured INTEGER,
    lineup_locked INTEGER,
    percent_owned REAL,
    proj_points REAL,           -- league-scored weekly projection (statSourceId=1)
    proj_stats_json TEXT,
    PRIMARY KEY (league_id, snapshot_id, player_id)
);

CREATE TABLE IF NOT EXISTS lineups (
    league_id INTEGER NOT NULL,
    snapshot_id INTEGER NOT NULL,
    scoring_period INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    player_id INTEGER NOT NULL,
    lineup_slot_id INTEGER,
    actual_points REAL,
    proj_points REAL,
    PRIMARY KEY (league_id, snapshot_id, scoring_period, team_id, player_id)
);

CREATE TABLE IF NOT EXISTS matchups (
    league_id INTEGER NOT NULL,
    snapshot_id INTEGER NOT NULL,
    matchup_period INTEGER NOT NULL,
    matchup_id INTEGER NOT NULL,
    home_team_id INTEGER,
    away_team_id INTEGER,
    home_points REAL,
    away_points REAL,
    home_points_live REAL,
    away_points_live REAL,
    home_proj_live REAL,        -- ESPN's live projected total
    away_proj_live REAL,
    home_win_prob REAL,         -- ESPN's own win probability: a baseline to beat
    away_win_prob REAL,
    PRIMARY KEY (league_id, snapshot_id, matchup_id)
);

CREATE TABLE IF NOT EXISTS odds_lines (
    snapshot_id INTEGER NOT NULL,
    event_id TEXT NOT NULL,
    commence_time TEXT,
    home_team TEXT,
    away_team TEXT,
    bookmaker TEXT NOT NULL,
    last_update TEXT,
    market TEXT NOT NULL,
    outcome TEXT NOT NULL,
    price REAL,
    point REAL
);
CREATE INDEX IF NOT EXISTS odds_snapshot ON odds_lines(snapshot_id);

CREATE TABLE IF NOT EXISTS odds_ledger (
    fetched_at TEXT NOT NULL,
    snapshot_id INTEGER,
    status_code INTEGER,
    credits_last INTEGER,
    credits_used INTEGER,
    credits_remaining INTEGER,
    kind TEXT                   -- odds (game lines) | props
);

CREATE TABLE IF NOT EXISTS prop_lines (
    snapshot_id INTEGER NOT NULL,
    event_id TEXT NOT NULL,
    commence_time TEXT,
    home_team TEXT,
    away_team TEXT,
    bookmaker TEXT NOT NULL,
    last_update TEXT,
    market TEXT NOT NULL,       -- player_anytime_td, player_receptions, ...
    player_name TEXT NOT NULL,  -- as the book spells it; not yet mapped to ESPN ids
    outcome TEXT NOT NULL,      -- Over | Under | Yes
    price REAL,
    point REAL
);
CREATE INDEX IF NOT EXISTS prop_lines_snapshot ON prop_lines(snapshot_id);

CREATE TABLE IF NOT EXISTS pro_games (
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    game_id INTEGER NOT NULL,
    kickoff_utc TEXT NOT NULL,
    home_pro_team_id INTEGER,
    away_pro_team_id INTEGER,
    refreshed_at TEXT NOT NULL,
    PRIMARY KEY (season, game_id)
);

CREATE TABLE IF NOT EXISTS game_context (
    snapshot_id INTEGER NOT NULL,
    game_id INTEGER NOT NULL,       -- ESPN pro game id (pro_games.game_id)
    week INTEGER,
    kickoff_utc TEXT,
    home TEXT, away TEXT,           -- nflverse abbreviations
    stadium TEXT,
    roof TEXT,                      -- outdoors | closed | dome (model vocabulary)
    temp_f REAL, wind_mph REAL, gust_mph REAL, precip_in REAL, precip_prob REAL,  -- forecast, outdoors only
    PRIMARY KEY (snapshot_id, game_id)
);

CREATE TABLE IF NOT EXISTS injury_reports (
    snapshot_id INTEGER NOT NULL,
    season INTEGER, week INTEGER, team TEXT, gsis_id TEXT, position TEXT, full_name TEXT,
    report_status TEXT, practice_status TEXT, injury TEXT
);
CREATE INDEX IF NOT EXISTS injury_reports_snapshot ON injury_reports(snapshot_id);

CREATE TABLE IF NOT EXISTS scheduled_wakes (
    wake_local TEXT PRIMARY KEY,
    slot_key TEXT,
    created_at TEXT
);
"""


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    _migrate(con)
    return con


def _migrate(con) -> None:
    _migrate_league_id(con)
    cols = {r["name"] for r in con.execute("PRAGMA table_info(odds_ledger)")}
    if "kind" not in cols:
        con.execute("ALTER TABLE odds_ledger ADD COLUMN kind TEXT")
        con.execute("UPDATE odds_ledger SET kind='odds'")
        con.commit()


def save_raw(con, snapshot_id, source, kind, url, params, status_code, body) -> int:
    data = json.dumps(body, separators=(",", ":")).encode() if body is not None else b""
    league = LEAGUE_ID if source == "espn" and kind in LEAGUE_KINDS else None
    cur = con.execute(
        "INSERT INTO raw_responses (snapshot_id, source, kind, url, params_json, fetched_at,"
        " status_code, sha256, body_gz, league_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (snapshot_id, source, kind, url, json.dumps(params, sort_keys=True), iso(utcnow()),
         status_code, hashlib.sha256(data).hexdigest(), gzip.compress(data), league),
    )
    con.commit()
    return cur.lastrowid


def load_raw(row) -> object:
    data = gzip.decompress(row["body_gz"])
    return json.loads(data) if data else None


def _migrate_league_id(con) -> None:
    """Add league_id to databases created before it existed. Every earlier row is this league's.

    One transaction under a write lock: concurrent connects (the launchd tick) wait, then see it done.
    """
    def has(t):
        return "league_id" in {r["name"] for r in con.execute(f"PRAGMA table_info({t})")}
    if has("raw_responses") and all(has(t) for t in LEAGUE_TABLES):
        return
    con.execute("BEGIN IMMEDIATE")
    try:
        if not has("raw_responses"):
            con.execute("ALTER TABLE raw_responses ADD COLUMN league_id INTEGER")
            con.execute(f"UPDATE raw_responses SET league_id=? WHERE source='espn' AND kind IN ({','.join('?' * len(LEAGUE_KINDS))})",
                        (LEAGUE_ID, *LEAGUE_KINDS))
        for t in LEAGUE_TABLES:  # league_id joins the primary key: rebuild the table and copy rows over
            if has(t):
                continue
            cols = [r["name"] for r in con.execute(f"PRAGMA table_info({t})")]
            con.execute(f"ALTER TABLE {t} RENAME TO {t}_old")
            create = SCHEMA[SCHEMA.index(f"CREATE TABLE IF NOT EXISTS {t} ("):]
            con.execute(create[:create.index(");") + 2])
            con.execute(f"INSERT INTO {t} (league_id, {', '.join(cols)}) SELECT ?, {', '.join(cols)} FROM {t}_old", (LEAGUE_ID,))
            con.execute(f"DROP TABLE {t}_old")
        con.execute("CREATE INDEX IF NOT EXISTS raw_league ON raw_responses(league_id, kind)")
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
