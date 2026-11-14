"""Trade analyzer: how each completed trade has played out so far.

Trades come from ESPN's archived transaction log (executed TRADE_ACCEPT entries).
For each trade and each finished week after it:
  got   the incoming players: points while on your roster, and as your starters
  gave  the outgoing players (plus anyone dropped to make room): points wherever they went
  undo  your highest-projection lineup with and without the trade (incoming players removed,
        outgoing and dropped players back), both scored on actual results against the
        opponent's actual score, so the trade's effect is isolated from your lineup choices.
The undo assumes you'd have kept the outgoing players and made every other move the same.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import polars as pl

from ..config import DB_PATH, LEAGUE_ID, MY_TEAM_ID, SEASON
from ..db import load_raw
from .backtest import _final_rosters
from .optimize import BENCH, IR, _slot_counts
from .report import _roster_names, team_names
from .review import FINISHED, _frame

FLEX_OK = ("RB", "WR", "TE")


def _con():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def _week_of(con, ms: int) -> int:
    """Scoring period a trade at this time first counts for: the first week whose last game is after it."""
    when = datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat()
    r = con.execute("SELECT MIN(week) FROM (SELECT week, MAX(kickoff_utc) k FROM pro_games WHERE season=? GROUP BY week)"
                    " WHERE k >= ?", (SEASON, when)).fetchone()
    return r[0] or 99


def executed_trades(con=None) -> list[dict]:
    """Every completed trade in the league, oldest first.

    From the league activity feed (all teams); players dropped to make room come from the
    transaction log, which details only your own team's trades.
    """
    con = con or _con()
    drops: dict[frozenset, list] = {}
    for r in con.execute("SELECT * FROM raw_responses WHERE kind='transactions' AND status_code=200 AND league_id=?", (LEAGUE_ID,)):
        for t in load_raw(r).get("transactions", []):
            if t.get("type") == "TRADE_ACCEPT" and t.get("status") == "EXECUTED" and t.get("items"):
                key = frozenset((i["playerId"], i["fromTeamId"], i["toTeamId"]) for i in t["items"] if i["type"] == "TRADE")
                drops[key] = [{"type": "DROP", "player_id": i["playerId"], "from": i["fromTeamId"], "to": None}
                              for i in t["items"] if i["type"] == "DROP"]
    seen: dict[str, dict] = {}
    for r in con.execute("SELECT * FROM raw_responses WHERE kind='trades' AND status_code=200 AND league_id=? ORDER BY id",
                         (LEAGUE_ID,)):
        for t in load_raw(r).get("topics", []):
            ms = [m for m in t.get("messages", []) if m.get("messageTypeId") == 244]
            if ms and not t.get("isDeleted"):
                seen[t["id"]] = {"date_ms": t["date"], "moves": [(m["targetId"], m["from"], m["to"]) for m in ms]}
    out = []
    for tid, t in sorted(seen.items(), key=lambda kv: kv[1]["date_ms"]):
        items = [{"type": "TRADE", "player_id": p, "from": f, "to": to} for p, f, to in t["moves"]]
        items += drops.get(frozenset(t["moves"]), [])
        out.append({"id": tid, "week": _week_of(con, t["date_ms"]),
                    "teams": sorted({i["from"] for i in items if i["type"] == "TRADE"} | {i["to"] for i in items if i["type"] == "TRADE"}),
                    "date": datetime.fromtimestamp(t["date_ms"] / 1000, timezone.utc).isoformat(), "items": items})
    return out


def _best_lineup(players: list[dict], counts: dict) -> list[dict]:
    """Highest-projection legal lineup (fixed slots by projection, then FLEX from what's left)."""
    left = sorted(players, key=lambda p: -(p["proj"] or 0))
    chosen = []
    for slot, n in counts.items():
        if slot == "FLEX":
            continue
        pick = [p for p in left if p["pos"] == slot][:n]
        chosen += pick
        left = [p for p in left if p not in pick]
    flex = [p for p in left if p["pos"] in FLEX_OK][:counts.get("FLEX", 0)]
    return chosen + flex


def analyze(team_id: int | None = None) -> dict:
    con = _con()
    me = team_id or MY_TEAM_ID
    counts = _slot_counts(con)
    names = team_names()
    frame = _frame()
    weeks = list(FINISHED)
    ros = {w: _final_rosters(con, w) for w in weeks}
    pinfo: dict[int, dict] = {}
    for w in weeks:
        for pid, (nm, pos, _) in _roster_names(con, "replay", None, w).items():
            pinfo.setdefault(pid, {"name": nm, "pos": pos})
    for r in frame.select("espn_id", "name", "pos").unique("espn_id").iter_rows(named=True):
        pinfo.setdefault(r["espn_id"], {"name": r["name"], "pos": r["pos"]})
    # per player-week: ESPN projection, actual, and where he was rostered / started
    pw = {(r["espn_id"], r["week"]): {"proj": r["proj"] or 0.0, "actual": r["actual"] or 0.0}
          for r in frame.select("espn_id", "week", "proj", "actual").iter_rows(named=True)}
    where = {}  # (pid, week) -> (team_id, started, espn actual)
    for w, df in ros.items():
        for r in df.iter_rows(named=True):
            where[(r["player_id"], w)] = (r["team_id"], r["slot"] not in (BENCH, IR), r["actual"] or 0.0)

    def pts(pid, w):
        loc = where.get((pid, w))
        return loc[2] if loc else pw.get((pid, w), {}).get("actual", 0.0)

    def opp_score(w, me):
        df = ros[w]
        opp = df.filter(pl.col("team_id") == me)["opp_team_id"][0]
        return float(df.filter((pl.col("team_id") == opp) & ~pl.col("slot").is_in([BENCH, IR]))["actual"].sum()), opp

    def lineup_pts(pids, w):
        players = [{"id": p, "pos": pinfo.get(p, {}).get("pos"), "proj": pw.get((p, w), {}).get("proj", 0.0)} for p in pids]
        best = _best_lineup(players, counts)
        return sum(pts(p["id"], w) for p in best), {p["id"] for p in best}

    # this week's ESPN projection (latest live snapshot), to show how each side looks now
    sid = con.execute("SELECT MAX(snapshot_id) FROM player_snapshots WHERE league_id=?", (LEAGUE_ID,)).fetchone()[0]
    now_week = con.execute("SELECT week FROM snapshots WHERE id=?", (sid,)).fetchone()[0] if sid else None
    now_proj = {r[0]: r[1] for r in con.execute("SELECT player_id, proj_points FROM player_snapshots WHERE league_id=? AND snapshot_id=?",
                                                (LEAGUE_ID, sid))}

    trades = executed_trades(con)

    def one(t, me):
        """How this trade has played out for team `me`."""
        partner = next(x for x in t["teams"] if x != me)
        got = [i["player_id"] for i in t["items"] if i["type"] == "TRADE" and i["to"] == me]
        gave = [i["player_id"] for i in t["items"] if i["type"] == "TRADE" and i["from"] == me]
        dropped = [i["player_id"] for i in t["items"] if i["type"] == "DROP" and i["from"] == me]
        # weeks the trade was in effect: from the first finished week an incoming player is on your roster,
        # until none of them is (traded on or dropped); a later trade of theirs is analysed on its own
        after = []
        for w in (w for w in weeks if w >= t["week"]):
            if any(where.get((p, w), (None,))[0] == me for p in got):
                after.append(w)
            elif after:
                break
        start = after[0] if after else None
        later = next((u for u in trades if u["week"] > t["week"] and
                      any(i["player_id"] in got and i["from"] == me for i in u["items"])), None)

        def side(pids, mine: bool):
            rows = []
            for p in pids:
                wk = []
                for w in after:
                    loc = where.get((p, w))
                    on_me = loc is not None and loc[0] == me
                    if mine and not on_me:
                        continue
                    proj = pw.get((p, w), {}).get("proj", 0.0)
                    wk.append({"week": w, "team": loc[0] if loc else None, "started": bool(loc and loc[1]),
                               "points": pts(p, w), "proj": proj, "no_game": not proj and not pts(p, w)})
                rows.append({"id": p, **pinfo.get(p, {"name": "?", "pos": None}),
                             "weeks": wk, "total": sum(x["points"] for x in wk),
                             "started_pts": sum(x["points"] for x in wk if x["started"]),
                             "started_weeks": sum(x["started"] for x in wk),
                             "now_with": names.get(where.get((p, after[-1]), (None,))[0]) if after else None,
                             "proj_now": now_proj.get(p)})
            return rows

        weekly = []
        for w in after:
            roster = [r["player_id"] for r in ros[w].filter(pl.col("team_id") == me).iter_rows(named=True)]
            undo = [p for p in roster if p not in got] + [p for p in gave + dropped if p not in roster]
            with_pts, with_ids = lineup_pts(roster, w)
            undo_pts, undo_ids = lineup_pts(undo, w)
            opp_pts, opp_id = opp_score(w, me)
            actual = float(ros[w].filter((pl.col("team_id") == me) & ~pl.col("slot").is_in([BENCH, IR]))["actual"].sum())
            weekly.append({"week": w, "opponent": names.get(opp_id), "opp_points": opp_pts, "actual": actual,
                           "with_trade": with_pts, "undone": undo_pts, "diff": with_pts - undo_pts,
                           "won_with": with_pts > opp_pts, "won_undone": undo_pts > opp_pts,
                           "got_started": [pinfo.get(p, {}).get("name") for p in got if p in with_ids],
                           "gave_would_start": [pinfo.get(p, {}).get("name") for p in gave + dropped if p in undo_ids]})
        flips = [x for x in weekly if x["won_with"] != x["won_undone"]]
        g, gv = side(got, True), side(gave, False)
        return {"id": t["id"], "week": t["week"], "date": t["date"], "partner": names.get(partner),
                    "got": g, "gave": gv, "dropped": side(dropped, False), "in_effect_from": start,
                    "in_effect_to": after[-1] if after else None, "ended_by_trade_week": later["week"] if later else None,
                    "weekly": weekly, "net_lineup": sum(x["diff"] for x in weekly),
                    "wins_gained": sum(x["won_with"] and not x["won_undone"] for x in flips),
                    "wins_lost": sum(x["won_undone"] and not x["won_with"] for x in flips),
                    "got_points": sum(p["total"] for p in g), "gave_points": sum(p["total"] for p in gv)}

    out = [one(t, me) for t in trades if me in t["teams"]]
    # every trade in the league, both sides (multi-team trades: each team vs. undoing its own part)
    league = []
    for t in reversed(trades):
        sides = [one(t, team) for team in t["teams"]]
        league.append({"id": t["id"], "week": t["week"], "date": t["date"], "involves_me": me in t["teams"],
                       "sides": [{"team": names.get(team), "got": [p["name"] for p in x["got"]], "gave": [p["name"] for p in x["gave"]],
                                  "net_lineup": x["net_lineup"], "wins_gained": x["wins_gained"], "wins_lost": x["wins_lost"],
                                  "weeks": len(x["weekly"])} for team, x in zip(t["teams"], sides)]})
    return {"team": {"id": me, "name": names.get(me)}, "weeks": weeks, "now_week": now_week, "trades": out, "league": league,
            "totals": {"net_lineup": sum(t["net_lineup"] for t in out),
                       "wins_gained": sum(t["wins_gained"] for t in out), "wins_lost": sum(t["wins_lost"] for t in out)}}
