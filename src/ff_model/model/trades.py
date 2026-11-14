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

from ..config import DB_PATH, MY_TEAM_ID
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


def executed_trades(con=None) -> list[dict]:
    """Every executed trade in the archived transaction log, oldest first."""
    con = con or _con()
    seen: dict[str, dict] = {}
    for r in con.execute("SELECT * FROM raw_responses WHERE kind='transactions' AND status_code=200 ORDER BY id"):
        for t in load_raw(r).get("transactions", []):
            if t.get("type") == "TRADE_ACCEPT" and t.get("status") == "EXECUTED" and t.get("items"):
                seen[t["id"]] = t
    out = []
    for t in sorted(seen.values(), key=lambda t: t.get("proposedDate") or 0):
        teams = sorted({i["fromTeamId"] for i in t["items"] if i["type"] == "TRADE"}
                       | {i["toTeamId"] for i in t["items"] if i["type"] == "TRADE"})
        out.append({"id": t["id"], "week": t["scoringPeriodId"], "teams": teams,
                    "date": datetime.fromtimestamp(t["proposedDate"] / 1000, timezone.utc).isoformat() if t.get("proposedDate") else None,
                    "items": [{"type": i["type"], "player_id": i["playerId"], "from": i["fromTeamId"], "to": i["toTeamId"]}
                              for i in t["items"]]})
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

    def opp_score(w):
        df = ros[w]
        opp = df.filter(pl.col("team_id") == me)["opp_team_id"][0]
        return float(df.filter((pl.col("team_id") == opp) & ~pl.col("slot").is_in([BENCH, IR]))["actual"].sum()), opp

    def lineup_pts(pids, w):
        players = [{"id": p, "pos": pinfo.get(p, {}).get("pos"), "proj": pw.get((p, w), {}).get("proj", 0.0)} for p in pids]
        best = _best_lineup(players, counts)
        return sum(pts(p["id"], w) for p in best), {p["id"] for p in best}

    # this week's ESPN projection (latest live snapshot), to show how each side looks now
    sid = con.execute("SELECT MAX(snapshot_id) FROM player_snapshots").fetchone()[0]
    now_week = con.execute("SELECT week FROM snapshots WHERE id=?", (sid,)).fetchone()[0] if sid else None
    now_proj = {r[0]: r[1] for r in con.execute("SELECT player_id, proj_points FROM player_snapshots WHERE snapshot_id=?", (sid,))}

    out = []
    for t in executed_trades(con):
        if me not in t["teams"]:
            continue
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
        later = next((u for u in executed_trades(con) if u["week"] > t["week"] and
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
            opp_pts, opp_id = opp_score(w)
            actual = float(ros[w].filter((pl.col("team_id") == me) & ~pl.col("slot").is_in([BENCH, IR]))["actual"].sum())
            weekly.append({"week": w, "opponent": names.get(opp_id), "opp_points": opp_pts, "actual": actual,
                           "with_trade": with_pts, "undone": undo_pts, "diff": with_pts - undo_pts,
                           "won_with": with_pts > opp_pts, "won_undone": undo_pts > opp_pts,
                           "got_started": [pinfo.get(p, {}).get("name") for p in got if p in with_ids],
                           "gave_would_start": [pinfo.get(p, {}).get("name") for p in gave + dropped if p in undo_ids]})
        flips = [x for x in weekly if x["won_with"] != x["won_undone"]]
        g, gv = side(got, True), side(gave, False)
        out.append({"id": t["id"], "week": t["week"], "date": t["date"], "partner": names.get(partner),
                    "got": g, "gave": gv, "dropped": side(dropped, False), "in_effect_from": start,
                    "in_effect_to": after[-1] if after else None, "ended_by_trade_week": later["week"] if later else None,
                    "weekly": weekly, "net_lineup": sum(x["diff"] for x in weekly),
                    "wins_gained": sum(x["won_with"] and not x["won_undone"] for x in flips),
                    "wins_lost": sum(x["won_undone"] and not x["won_with"] for x in flips),
                    "got_points": sum(p["total"] for p in g), "gave_points": sum(p["total"] for p in gv)})
    return {"team": {"id": me, "name": names.get(me)}, "weeks": weeks, "now_week": now_week, "trades": out,
            "totals": {"net_lineup": sum(t["net_lineup"] for t in out),
                       "wins_gained": sum(t["wins_gained"] for t in out), "wins_lost": sum(t["wins_lost"] for t in out)}}
