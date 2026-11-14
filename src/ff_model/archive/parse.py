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
    for t in ("player_snapshots", "lineups", "matchups", "odds_lines", "prop_lines",
              "game_context", "injury_reports"):
        con.execute(f"DELETE FROM {t} WHERE snapshot_id=?", (snapshot_id,))
    rows = con.execute("SELECT * FROM raw_responses WHERE snapshot_id=? AND status_code=200",
                       (snapshot_id,)).fetchall()
    for row in rows:
        body = load_raw(row)
        if row["kind"] == "players":
            _players(con, row["league_id"], snapshot_id, week, body)
        elif row["kind"] in ("league", "prev_week"):
            period = json.loads(row["params_json"])["scoringPeriodId"]
            _league(con, row["league_id"], snapshot_id, period, body)
        elif row["kind"] == "odds":
            _odds(con, snapshot_id, body)
        elif row["kind"] == "props":
            _props(con, snapshot_id, body)
        elif row["kind"] == "weather":
            _game_context(con, snapshot_id, json.loads(row["params_json"]), body)
        elif row["kind"] == "nflverse_injuries":
            _injuries(con, snapshot_id, body)
    con.commit()


def _players(con, league_id, sid, week, body):
    for w in body.get("players", []):
        p = w.get("player", {})
        proj = _weekly(p.get("stats"), 1, week)
        con.execute(
            "INSERT OR REPLACE INTO player_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (league_id, sid, w["id"], p.get("fullName"), p.get("defaultPositionId"), p.get("proTeamId"),
             w.get("status"), w.get("onTeamId"), p.get("injuryStatus"), int(bool(p.get("injured"))),
             int(bool(w.get("lineupLocked"))), (p.get("ownership") or {}).get("percentOwned"),
             proj.get("appliedTotal") if proj else None,
             json.dumps(proj.get("stats")) if proj else None))


def _league(con, league_id, sid, period, body):
    for t in body.get("teams", []):
        for e in (t.get("roster") or {}).get("entries", []):
            stats = ((e.get("playerPoolEntry") or {}).get("player") or {}).get("stats")
            act, proj = _weekly(stats, 0, period), _weekly(stats, 1, period)
            con.execute("INSERT OR REPLACE INTO lineups VALUES (?,?,?,?,?,?,?,?)",
                        (league_id, sid, period, t["id"], e["playerId"], e.get("lineupSlotId"),
                         act.get("appliedTotal") if act else None,
                         proj.get("appliedTotal") if proj else None))
    for m in body.get("schedule", []):
        if m.get("matchupPeriodId") != period:
            continue
        home, away = m.get("home") or {}, m.get("away") or {}
        con.execute("INSERT OR REPLACE INTO matchups VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (league_id, sid, period, m["id"], home.get("teamId"), away.get("teamId"),
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


def _props(con, sid, ev):
    for bk in (ev or {}).get("bookmakers", []):
        for mk in bk.get("markets", []):
            for oc in mk.get("outcomes", []):
                con.execute("INSERT INTO prop_lines VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                            (sid, ev["id"], ev.get("commence_time"), ev.get("home_team"),
                             ev.get("away_team"), bk["key"], mk.get("last_update"), mk["key"],
                             oc.get("description"), oc["name"], oc.get("price"), oc.get("point")))


def _game_context(con, sid, params, body):
    """One row per upcoming game (all roof types); forecast columns for outdoor games."""
    from datetime import datetime

    from ..sources import weather
    games = params["games"]
    outdoor = [g for g in games if g["roof"] == "outdoors"]
    for g in outdoor:
        g["kickoff"] = datetime.fromisoformat(g["kickoff_utc"])
    fc = {w["game_id"]: w for w in (weather.summarize(body, outdoor) if body and outdoor else [])}
    for g in games:
        w = fc.get(g["game_id"], {})
        con.execute("INSERT OR REPLACE INTO game_context VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (sid, g["game_id"], g["week"], g["kickoff_utc"], g["home"], g["away"], g["stadium"],
                     g["roof"], w.get("temp_f"), w.get("wind_mph"), w.get("gust_mph"),
                     w.get("precip_in"), w.get("precip_prob")))


def _injuries(con, sid, rows):
    for r in rows or []:
        con.execute("INSERT INTO injury_reports VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (sid, r.get("season"), r.get("week"), r.get("team"), r.get("gsis_id"), r.get("position"),
                     r.get("full_name"), r.get("report_status"), r.get("practice_status"),
                     r.get("report_primary_injury")))


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
