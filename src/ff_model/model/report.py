"""Everything the recommendation page shows, as one JSON-serializable dict.

Built from one archived snapshot (latest by default), so the page reflects
exactly what the model knew at that moment.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime

import numpy as np
import polars as pl

from ..config import DB_PATH, MY_TEAM_ID
from ..db import load_raw
from . import correlation as C
from .live import TEAM_NAMES, build_inputs
from .marginals import MODEL_PATH, TAUS, Marginals
from .optimize import BENCH, IR, SLOT, _opponent_starters, _reason, _rosters, _slot_counts, decide

SHOW_TAUS = (0.1, 0.25, 0.5, 0.75, 0.9)
# input columns to neutralize per factor group, to show what each one did to the median
GROUPS = {
    "weather": {"roof": None, "wind": None, "temp": None},
    "spread": {"exp_margin": None},
    "injury": {"report_status": "none", "practice_status": None, "games_back": 0},
    "lines": {"ol_out": 0, "opp_ol_out": 0, "opp_db_out": 0},
}


def _neutral(df: pl.DataFrame, cols: dict) -> pl.DataFrame:
    return df.with_columns([pl.lit(v).cast(df.schema[c]).alias(c) if v is not None else pl.lit(None).cast(df.schema[c]).alias(c)
                            for c, v in cols.items() if c in df.columns])


def _team_names(con, sid: int) -> dict[int, str]:
    row = con.execute("SELECT * FROM raw_responses WHERE kind='league' AND status_code=200 AND snapshot_id<=?"
                      " ORDER BY id DESC LIMIT 1", (sid,)).fetchone()
    return {t["id"]: (t.get("name") or f"Team {t['id']}").strip() for t in load_raw(row)["teams"]}


def _freshness(con, sid: int) -> dict:
    def t(sql):
        r = con.execute(sql, (sid,)).fetchone()
        return r[0] if r else None
    return {
        "snapshot": t("SELECT started_at FROM snapshots WHERE id=?"),
        "odds": t("SELECT fetched_at FROM raw_responses WHERE kind='odds' AND status_code=200 AND snapshot_id<=? ORDER BY id DESC LIMIT 1"),
        "weather": t("SELECT fetched_at FROM raw_responses WHERE kind='weather' AND snapshot_id<=? ORDER BY id DESC LIMIT 1"),
        "injuries": t("SELECT fetched_at FROM raw_responses WHERE kind='nflverse_injuries' AND snapshot_id<=? ORDER BY id DESC LIMIT 1"),
    }


def _totals(con, sid: int) -> dict[str, float]:
    """Game total (median across books) per team, latest odds at or before the snapshot."""
    osid = con.execute("SELECT MAX(snapshot_id) FROM odds_lines WHERE snapshot_id <= ?", (sid,)).fetchone()[0]
    if not osid:
        return {}
    rows = con.execute("SELECT home_team, away_team, commence_time, point FROM odds_lines WHERE snapshot_id=?"
                       " AND market='totals' AND outcome='Over'", (osid,)).fetchall()
    by: dict[tuple, list] = {}
    for h, a, ct, pt in rows:
        by.setdefault((TEAM_NAMES.get(h), TEAM_NAMES.get(a), ct), []).append(pt)
    out = {}
    for (h, a, ct), pts in sorted(by.items(), key=lambda kv: kv[0][2]):
        for team in (h, a):
            out.setdefault((team, ct[:10]), float(np.median(pts)))
    return out


def build_report(snapshot_id: int | None = None, n: int = 20000, seed: int = 7) -> dict:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    pool = C.assign_roles(build_inputs(snapshot_id))
    sid, week = int(pool["snapshot_id"][0]), int(pool["week"][0])
    taken = datetime.fromisoformat(con.execute("SELECT started_at FROM snapshots WHERE id=?", (sid,)).fetchone()[0])
    counts = _slot_counts(con)
    opp_id, ros = _rosters(con, sid, week)
    df = ros.join(pool, on="player_id", how="left").with_columns(
        pl.col("proj").fill_null(0.0), pl.col("name").fill_null("?"),
        (pl.col("kickoff_utc").str.to_datetime(time_zone="UTC") <= pl.lit(taken)).fill_null(False).alias("locked"),
        (pl.col("report_status").is_in(["Out"]) | (pl.col("slot") == IR) | pl.col("opp").is_null()).fill_null(True)
        .alias("unavailable"))
    marg = Marginals.load(MODEL_PATH)
    dfq = df.with_columns(pl.col("pos").fill_null("WR"))
    q = marg.quantiles(dfq)
    taus = list(TAUS)
    med = q[:, taus.index(0.5)]
    effects = {g: med - marg.quantiles(_neutral(dfq, cols))[:, taus.index(0.5)] for g, cols in GROUPS.items()}

    opp_start = _opponent_starters(df.filter(pl.col("team_id") == opp_id), counts)
    mine = df.filter(pl.col("team_id") == MY_TEAM_ID)
    q_of = dict(zip(df["player_id"].to_list(), q))
    dec = decide(mine, opp_start, q_of, counts, C.load(), n=n, seed=seed)
    best_ids = {p for _, p in dec.lineups[dec.best]}
    best_slot = {p: s for s, p in dec.lineups[dec.best]}
    mp_ids = {p for _, p in dec.lineups[dec.max_proj]}
    member = [{p for _, p in lu} for lu in dec.lineups]

    # correlation notes need games of my starters and the opponent's starters
    my_start_rows = mine.filter(pl.col("player_id").is_in(list(best_ids)) | (pl.col("locked") & ~pl.col("slot").is_in([BENCH, IR])))
    my_games = {(r["game"], r["team"], r["pos"], r["name"]) for r in my_start_rows.iter_rows(named=True) if r["game"]}
    opp_games = {(r["game"], r["team"], r["pos"], r["name"]) for r in opp_start.iter_rows(named=True) if r["game"]}
    totals = _totals(con, sid)

    def player(i: int, r: dict, side: str) -> dict:
        p = r["player_id"]
        out = {
            "id": p, "name": r["name"], "pos": r["pos"], "team": r["team"], "opp": r["opp"], "kickoff": r["kickoff_utc"],
            "slot": SLOT.get(r["slot"], "BE" if r["slot"] == BENCH else "IR"), "proj": r["proj"], "locked": r["locked"],
            "actual": r["actual"], "unavailable": r["unavailable"] and not r["locked"], "role": r["role"],
            "status": r["report_status"], "practice": r["practice_status"], "injury": r["injury"],
            "games_back": r["games_back"], "roof": r["roof"], "wind": r["wind"], "temp": r["temp"],
            "gust": r.get("gust_mph"), "precip_prob": r.get("precip_prob"), "spread": r["exp_margin"],
            "ol_out": r["ol_out"], "opp_ol_out": r["opp_ol_out"], "opp_db_out": r["opp_db_out"],
            "quantiles": {f"p{int(t*100)}": float(q[i, taus.index(t)]) for t in SHOW_TAUS},
            "effects": {g: float(effects[g][i]) for g in GROUPS},
        }
        if r["kickoff_utc"] and r["team"]:
            out["total"] = totals.get((r["team"], r["kickoff_utc"][:10]))
        if p in (dec.col or {}):
            x = dec.D[:, dec.col[p]]
            out["chance"] = {"beat_proj": float((x > r["proj"]).mean()), "dud": float((x < 5).mean()),
                             "twenty": float((x >= 20).mean()), "in_p10_p90": 0.8}
        if side == "me" and not r["locked"] and not out["unavailable"]:
            with_p = [dec.pwin[k] for k, m in enumerate(member) if p in m]
            without = [dec.pwin[k] for k, m in enumerate(member) if p not in m]
            out["recommended"] = p in best_ids
            out["rec_slot"] = best_slot.get(p)
            out["in_max_proj"] = p in mp_ids
            out["pwin_if_started"] = float(max(with_p)) if with_p else None
            out["pwin_if_benched"] = float(max(without)) if without else None
        if r["game"]:
            out["stacks"] = sorted(f"{nm} ({ps})" for g, t, ps, nm in my_games
                                   if g == r["game"] and t == r["team"] and nm != r["name"]) if side == "me" else []
            out["vs_opponent"] = sorted(f"{nm} ({ps}, {t})" for g, t, ps, nm in opp_games if g == r["game"]
                                        and nm != r["name"]) if side == "me" else []
        return out

    rows = list(df.iter_rows(named=True))
    me_rows = [player(i, r, "me") for i, r in enumerate(rows) if r["team_id"] == MY_TEAM_ID]
    opp_ids = set(opp_start["player_id"])
    opp_rows = [player(i, r, "opp") for i, r in enumerate(rows) if r["team_id"] == opp_id]
    for o in opp_rows:
        o["starting"] = o["id"] in opp_ids

    margin = dec.diff[:, dec.best]
    edges = np.arange(np.floor(np.percentile(margin, 0.5) / 5) * 5, np.ceil(np.percentile(margin, 99.5) / 5) * 5 + 5, 5)
    hist, _ = np.histogram(margin, bins=edges)
    order = np.argsort(-dec.pwin)
    info = {r["player_id"]: r["name"] for r in rows}
    best_names = {info[p] for p in best_ids if p in info}

    def alt(k):
        names = {info.get(p, "(empty)") for _, p in dec.lineups[k]}
        return {"pwin": float(dec.pwin[k]), "in": sorted(names - best_names), "out": sorted(best_names - names),
                "mean": float(dec.my_totals[:, k].mean())}

    matchup = con.execute("SELECT * FROM matchups WHERE snapshot_id=? AND matchup_period=? AND ? IN (home_team_id, away_team_id)",
                          (sid, week, MY_TEAM_ID)).fetchone()
    espn_wp = None
    if matchup:
        espn_wp = matchup["home_win_prob"] if matchup["home_team_id"] == MY_TEAM_ID else matchup["away_win_prob"]
    names = _team_names(con, sid)
    games = {}
    for r in me_rows + opp_rows:
        if r.get("kickoff") and r["team"]:
            key = tuple(sorted([r["team"], r["opp"]]))
            games[key] = {"teams": f"{r['team']} vs {r['opp']}", "kickoff": r["kickoff"], "roof": r["roof"],
                          "wind": r["wind"], "gust": r["gust"], "temp": r["temp"], "precip_prob": r["precip_prob"],
                          "total": r.get("total"), "spread_note": f"{r['team']} {-(r['spread'] or 0):+.1f}" if r["spread"] is not None else None,
                          "mine": sorted({x["name"] for x in me_rows if x["team"] in key and (x.get("recommended") or x["locked"])}),
                          "theirs": sorted({x["name"] for x in opp_rows if x["team"] in key and x.get("starting")})}
    by_id = {r["id"]: dict(r, p10=r["quantiles"]["p10"], p90=r["quantiles"]["p90"],
                           game=next((x["game"] for x in rows if x["player_id"] == r["id"]), None)) for r in me_rows}
    ins, outs = best_ids - mp_ids, mp_ids - best_ids
    reason = _reason(float(dec.pwin[dec.max_proj]), {p for p in ins if p in by_id}, {p for p in outs if p in by_id},
                     by_id, opp_start, df)
    return {
        "snapshot_id": sid, "week": week, "freshness": _freshness(con, sid), "reason": reason,
        "me": {"id": MY_TEAM_ID, "name": names.get(MY_TEAM_ID)}, "opponent": {"id": opp_id, "name": names.get(opp_id)},
        "p_win": float(dec.pwin[dec.best]), "p_win_max_proj": float(dec.pwin[dec.max_proj]), "espn_win_prob": espn_wp,
        "my_mean": float(dec.my_totals[:, dec.best].mean()), "opp_mean": float(dec.opp_total.mean()),
        "my_range": [float(np.percentile(dec.my_totals[:, dec.best], p)) for p in (10, 50, 90)],
        "opp_range": [float(np.percentile(dec.opp_total, p)) for p in (10, 50, 90)],
        "margin_hist": {"edges": edges.tolist(), "counts": hist.tolist(), "n": int(len(margin))},
        "changes": {"start": sorted(info[p] for p in best_ids - mp_ids if p in info),
                    "bench": sorted(info[p] for p in mp_ids - best_ids if p in info)},
        "close": bool(len(order) > 1 and dec.pwin[order[0]] - dec.pwin[order[1]] < 0.01),
        "alternatives": [alt(int(k)) for k in order[1:6]],
        "players": {"me": me_rows, "opp": opp_rows},
        "games": sorted(games.values(), key=lambda g: g["kickoff"]),
        "draws": n,
    }
