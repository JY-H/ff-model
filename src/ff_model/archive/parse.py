"""Derive tidy tables from archived raw responses (re-runnable)."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from ..db import iso, load_raw, utcnow


def _weekly(stats, source: int, week: int):
    for s in stats or []:
        if s.get("statSourceId") == source and s.get("statSplitTypeId") == 1 \
                and s.get("scoringPeriodId") == week:
            return s
    return None


def parse_snapshot(con, snapshot_id: int, week: int) -> None:
    for t in ("player_snapshots", "lineups", "matchups", "odds_lines"):
        con.execute(f"DELETE FROM {t} WHERE snapshot_id=?", (snapshot_id,))
    rows = con.execute("SELECT * FROM raw_responses WHERE snapshot_id=? AND status_code=200",
                       (snapshot_id,)).fetchall()
    for row in rows:
        body = load_raw(row)
        if row["kind"] == "players":
            _players(con, snapshot_id, week, body)
        elif row["kind"] in ("league", "prev_week"):
            period = json.loads(row["params_json"])["scoringPeriodId"]
            _league(con, snapshot_id, period, body)
        elif row["kind"] == "odds":
            _odds(con, snapshot_id, body)
    con.commit()


def _players(con, sid, week, body):
    for w in body.get("players", []):
        p = w.get("player", {})
        proj = _weekly(p.get("stats"), 1, week)
        con.execute(
            "INSERT OR REPLACE INTO player_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (sid, w["id"], p.get("fullName"), p.get("defaultPositionId"), p.get("proTeamId"),
             w.get("status"), w.get("onTeamId"), p.get("injuryStatus"), int(bool(p.get("injured"))),
             int(bool(w.get("lineupLocked"))), (p.get("ownership") or {}).get("percentOwned"),
             proj.get("appliedTotal") if proj else None,
             json.dumps(proj.get("stats")) if proj else None))


def _league(con, sid, period, body):
    for t in body.get("teams", []):
        for e in (t.get("roster") or {}).get("entries", []):
            stats = ((e.get("playerPoolEntry") or {}).get("player") or {}).get("stats")
            act, proj = _weekly(stats, 0, period), _weekly(stats, 1, period)
            con.execute("INSERT OR REPLACE INTO lineups VALUES (?,?,?,?,?,?,?)",
                        (sid, period, t["id"], e["playerId"], e.get("lineupSlotId"),
                         act.get("appliedTotal") if act else None,
                         proj.get("appliedTotal") if proj else None))
    for m in body.get("schedule", []):
        if m.get("matchupPeriodId") != period:
            continue
        home, away = m.get("home") or {}, m.get("away") or {}
        con.execute("INSERT OR REPLACE INTO matchups VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (sid, period, m["id"], home.get("teamId"), away.get("teamId"),
                     home.get("totalPoints"), away.get("totalPoints"),
                     home.get("totalPointsLive"), away.get("totalPointsLive"),
                     home.get("totalProjectedPointsLive"), away.get("totalProjectedPointsLive"),
                     home.get("winProbability"), away.get("winProbability")))


def _odds(con, sid, body):
    for ev in body or []:
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                for oc in mk.get("outcomes", []):
                    con.execute("INSERT INTO odds_lines VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                                (sid, ev["id"], ev.get("commence_time"), ev.get("home_team"),
                                 ev.get("away_team"), bk["key"], mk.get("last_update"),
                                 mk["key"], oc["name"], oc.get("price"), oc.get("point")))


def store_pro_games(con, season: int, body) -> int:
    now = iso(utcnow())
    games = {}
    for team in body["settings"]["proTeams"]:
        for wk, gs in (team.get("proGamesByScoringPeriod") or {}).items():
            for g in gs:
                games[g["id"]] = (int(wk), g)
    for gid, (wk, g) in games.items():
        kick = datetime.fromtimestamp(g["date"] / 1000, timezone.utc)
        con.execute("INSERT OR REPLACE INTO pro_games VALUES (?,?,?,?,?,?,?)",
                    (season, wk, gid, iso(kick), g.get("homeProTeamId"), g.get("awayProTeamId"), now))
    con.commit()
    return len(games)
