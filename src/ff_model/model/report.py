"""Everything the recommendation page shows, as JSON-serializable dicts.

Two sources, one shape:
  live   - weeks with archive snapshots: exactly what was known at that snapshot.
  replay - completed weeks before the archive existed (2026 W1-9): ESPN's
           projections (identical to what the league showed) plus nflverse
           context, decided as of Sunday 11:45 ET, with the final results shown.
Any team can be the "me" side.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
import polars as pl

from ..config import DATA_DIR, DB_PATH, MY_TEAM_ID, SEASON
from ..db import load_raw
from ..research.line_injuries import starters_out
from . import correlation as C
from .live import TEAM_NAMES, build_inputs
from .marginals import MODEL_PATH, TAUS, Marginals, anchor, curve_mean, espn_context_quantiles, features, neutralize
from .optimize import BENCH, IR, SLOT, _opponent_starters, _reason, _slot_counts, decide

SHOW_TAUS = (0.1, 0.25, 0.5, 0.75, 0.9)
GROUPS = {
    "weather": {"roof": None, "wind": None, "temp": None},
    "spread": {"exp_margin": None},
    "injury": {"report_status": "none", "practice_status": None, "games_back": 0},
    "lines": {"ol_out": 0, "opp_ol_out": 0, "opp_db_out": 0},
}
STAT_NAMES = {"0": "pass att", "1": "completions", "3": "pass yds", "4": "pass TD", "20": "INT", "23": "rush att",
              "24": "rush yds", "25": "rush TD", "58": "targets", "53": "receptions", "42": "rec yds", "43": "rec TD",
              "72": "fumbles lost", "83": "FG made", "84": "FG att", "86": "XP made", "99": "sacks",
              "95": "INT (def)", "96": "fumble rec", "120": "pts allowed", "127": "yds allowed"}


# --------------------------------------------------------------------------- inputs

def _con():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def _live_weeks(con) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for r in con.execute("SELECT s.id, s.week, s.slot_key, s.started_at FROM snapshots s WHERE s.status IN ('ok','partial')"
                         " AND (s.slot_key IS NULL OR s.slot_key NOT LIKE '%backfill')"
                         " AND EXISTS (SELECT 1 FROM player_snapshots p WHERE p.snapshot_id = s.id) ORDER BY s.id DESC"):
        out.setdefault(r["week"], []).append(dict(r))
    return out


def weeks_available() -> list[dict]:
    con = _con()
    live = _live_weeks(con)
    replay = {r[0] for r in con.execute("SELECT DISTINCT scoring_period FROM lineups WHERE actual_points IS NOT NULL")}
    hist = {int(p.stem[1:3]) for p in (DATA_DIR / "espn_history" / str(SEASON)).glob("w*-p0.json.gz")}
    weeks = []
    for w in sorted((replay & hist) | set(live)):
        weeks.append({"week": w, "source": "live" if w in live else "replay", "snapshots": live.get(w, [])})
    return weeks


def team_names(con=None, sid: int | None = None) -> dict[int, str]:
    con = con or _con()
    row = con.execute("SELECT * FROM raw_responses WHERE kind IN ('league','prev_week') AND status_code=200"
                      + (" AND snapshot_id<=?" if sid else "") + " ORDER BY id DESC LIMIT 1", (sid,) if sid else ()).fetchone()
    return {t["id"]: (t.get("name") or f"Team {t['id']}").strip() for t in load_raw(row)["teams"]}


@lru_cache(maxsize=1)
def _replay_frame() -> pl.DataFrame:
    from ..research.factor_tests import factor_frame
    from .backtest import _team_kickoffs
    f = factor_frame([SEASON], depth=50.0, with_u=False)
    k = _team_kickoffs(_con()).with_columns(pl.col("kickoff_utc").dt.strftime("%Y-%m-%dT%H:%M:%S+00:00"))
    return f.join(k, on=["week", "team"], how="left")


@lru_cache(maxsize=4)
def _hist_statlines(week: int) -> dict[int, dict]:
    from ..research.espn_history import parse
    d = parse([SEASON], write=False).filter(pl.col("week") == week)
    cols = [c for c in d.columns if c.startswith("st_")]
    return {r["espn_id"]: {c[3:]: r[c] for c in cols if r[c] is not None} for r in d.iter_rows(named=True)}


def _inputs(week: int | None, snapshot_id: int | None):
    """-> (source, sid, week, decision time, pool, rosters, statlines)"""
    con = _con()
    live = _live_weeks(con)
    if snapshot_id is None and week is None:
        week = max(set(live) | {w["week"] for w in weeks_available()})
    if snapshot_id is not None or week in live:
        sid = snapshot_id or live[week][0]["id"]
        pool = build_inputs(sid)
        week = int(pool["week"][0])
        taken = datetime.fromisoformat(con.execute("SELECT started_at FROM snapshots WHERE id=?", (sid,)).fetchone()[0])
        rows = con.execute("SELECT team_id, player_id, lineup_slot_id, actual_points FROM lineups WHERE snapshot_id=?"
                           " AND scoring_period=?", (sid, week)).fetchall()
        mus = con.execute("SELECT home_team_id, away_team_id FROM matchups WHERE snapshot_id=? AND matchup_period=?",
                          (sid, week)).fetchall()
        opp = {h: a for h, a in mus} | {a: h for h, a in mus}
        ros = pl.DataFrame([dict(team_id=r[0], player_id=r[1], slot=r[2], actual=r[3], final=None,
                                 opp_team_id=opp.get(r[0])) for r in rows])
        stat = {r[0]: json.loads(r[1] or "{}") for r in con.execute(
            "SELECT player_id, proj_stats_json FROM player_snapshots WHERE snapshot_id=?", (sid,))}
        return "live", sid, week, taken, C.assign_roles(pool), ros, stat
    from .backtest import _decision_time, _final_rosters
    f = _replay_frame().filter(pl.col("week") == week)
    pool = f.select(
        pl.col("espn_id").alias("player_id"), "name", "pos", "proj", "team", "opp", "season", "week",
        "report_status", "practice_status", "injury", "games_back", "roof", "wind", "temp", "exp_margin",
        "ol_out", "opp_ol_out", "opp_db_out", "kickoff_utc", "played",
        *[c for c in f.columns if c.startswith("p_")]).with_columns(
        pl.lit(None, dtype=pl.Float64).alias("gust_mph"), pl.lit(None, dtype=pl.Float64).alias("precip_prob"),
        pl.lit(None, dtype=pl.Int64).alias("on_team_id"))
    taken = _decision_time(con, week).astimezone(timezone.utc)
    ros = _final_rosters(con, week).with_columns(pl.col("actual").alias("final"))
    return "replay", None, week, taken, C.assign_roles(pool), ros, _hist_statlines(week)


# --------------------------------------------------------------------------- explanations

@lru_cache(maxsize=1)
def _evidence() -> pl.DataFrame | None:
    p = DATA_DIR / "factor_screen.csv"
    return pl.read_csv(p) if p.exists() else None


def _ev(factor: str, pos: str, group: str) -> str | None:
    e = _evidence()
    if e is None:
        return None
    r = e.filter((pl.col("factor") == factor) & (pl.col("pos") == pos) & (pl.col("group") == group))
    if r.height == 0:
        return None
    r = r.row(0, named=True)
    if abs(r["shift"]) <= r["ci95"]:
        return (f"2019–2025: no reliable difference for {pos}s in this situation ({group}): "
                f"{r['shift']:+.1f} ± {r['ci95']:.1f} vs. ESPN, n={r['n']}.")
    return (f"2019–2025: {pos}s in this situation ({group}) scored {r['shift']:+.1f} vs. ESPN relative to other {pos}s "
            f"(± {r['ci95']:.1f}, n={r['n']}, same direction in {r['seasons']} seasons).")


def _wind_group(p):
    if p["roof"] not in ("outdoors", "open"):
        return "dome/closed"
    w = p["wind"] or 0
    return "wind 15+" if w >= 15 else "wind 10-14" if w >= 10 else "wind <10"


def _explain(p: dict, effects: dict, shape: float, line_who: dict) -> list[dict]:
    pos = p["pos"]
    out = [{"factor": "shape", "effect": shape,
            "headline": "Where the median sits before any context",
            "details": [_shape_text(p, shape)]}]
    # weather
    w = effects["weather"]
    if p["roof"] in ("outdoors", "open") and p["wind"] is not None:
        wx = f"Outdoors. Forecast for the game window: {p['wind']:.0f} mph wind"
        wx += f" (gusts to {p['gust']:.0f})" if p.get("gust") else ""
        wx += f", {p['temp']:.0f}°F" if p["temp"] is not None else ""
        wx += f", {p['precip_prob']:.0f}% chance of rain." if p.get("precip_prob") is not None else "."
    elif p["roof"]:
        wx = "Dome." if p["roof"] == "dome" else "Retractable roof, assumed closed (teams close it in bad weather)."
    else:
        wx = "No stadium information."
    ev = [_ev("wind", pos, _wind_group(p))]
    if p["roof"] in ("outdoors", "open") and p["temp"] is not None and p["temp"] <= 45:
        ev.append(_ev("temperature", pos, "freezing (<=32F)" if p["temp"] <= 32 else "cold (33-45F)"))
    if pos == "RB":
        ev = ["The model doesn't adjust RBs for weather: the run game showed no weather effect."]
    out.append({"factor": "weather", "effect": w, "headline": "Weather and roof", "details": [wx],
                "evidence": [e for e in ev if e]})
    # spread
    if p["spread"] is not None:
        side = f"favored by {p['spread']:.1f}" if p["spread"] >= 0 else f"a {-p['spread']:.1f}-point underdog"
        grp = "fav 7+" if p["spread"] >= 7 else "dog 7+" if p["spread"] <= -7 else "within 7"
        det = [f"{p['team']} is {side} (median line across sportsbooks)" + (f"; game total {p['total']:.1f}." if p.get("total") else ".")]
        out.append({"factor": "spread", "effect": effects["spread"], "headline": "Betting line (game script)",
                    "details": det, "evidence": [e for e in [_ev("expected margin", pos, grp)] if e]})
    # injury
    det, ev = [], []
    if p["status"] and p["status"] != "none":
        det.append(f"Injury designation: {p['status']}" + (f" ({p['injury']})" if p.get("injury") else "") + ".")
        ev.append(_ev("injury designation", pos, "Q/D"))
    if p.get("practice") and p["status"] not in (None, "none"):
        det.append(f"Final practice report of the week: {p['practice']}.")
        pr = "DNP" if "Did Not" in p["practice"] else "limited" if "Limited" in p["practice"] else "full (listed)"
        ev.append(_ev("practice (played)", pos, pr))
    if p.get("games_back") in (1, 2, 3):
        det.append(f"Game {p['games_back']} back after missing 2+ games.")
        ev.append(_ev("return after 2+ missed", pos, ["1st game back", "2nd game back", "3rd game back"][p["games_back"] - 1]))
    if pos == "QB" and det:
        det.append("The model doesn't adjust QBs for injury status: too few listed QBs to learn from.")
    if det:
        out.append({"factor": "injury", "effect": effects["injury"], "headline": "Injury report",
                    "details": det, "evidence": [e for e in ev if e]})
    # lines
    det, ev = [], []
    own_ol, opp_ol, opp_db = line_who.get(("ol", p["team"]), ""), line_who.get(("ol", p["opp"]), ""), line_who.get(("db", p["opp"]), "")
    if pos in ("QB", "RB") and own_ol:
        det.append(f"{p['team']} offensive line starters out: {own_ol}.")
        ev.append(_ev("own OL starters out", pos, "2+ out" if (p["ol_out"] or 0) >= 2 else "1 out"))
    if pos == "D/ST" and opp_ol:
        det.append(f"Opponent ({p['opp']}) offensive line starters out: {opp_ol}.")
        ev.append(_ev("opp OL starters out", pos, "2+ out" if (p["opp_ol_out"] or 0) >= 2 else "1 out"))
    if pos == "QB" and opp_db:
        det.append(f"Opponent ({p['opp']}) secondary starters out: {opp_db}.")
        ev.append(_ev("opp DB starters out", pos, "2+ out" if (p["opp_db_out"] or 0) >= 2 else "1 out"))
    if det:
        out.append({"factor": "lines", "effect": effects["lines"], "headline": "Offensive line / secondary",
                    "details": det + ["Starters = 60%+ of snaps in earlier games; out = listed Out/Doubtful or on IR."],
                    "evidence": [e for e in ev if e]})
    return out


def _shape_text(p: dict, shape: float) -> str:
    pos, proj = p["pos"], p["proj"] or 0
    if not proj:
        return "No ESPN projection."
    td = p.get("td_share")
    rec = p.get("rec_share")
    txt = (f"Before any context, the model's median is {abs(shape):.1f} {'below' if shape < 0 else 'above'} "
           f"ESPN's {proj:.1f}. ")
    if pos in ("RB", "WR", "TE") and td is not None:
        txt += (f"{td:.0%} of the projection is expected touchdowns, which either happen or don't, so the typical week "
                f"lands {'below' if shape < 0 else 'near'} the average with a long upside tail")
        txt += f"; {rec:.2f} receptions per projected point add a steadier floor." if rec else "."
    elif pos == "QB":
        txt += "QB outcomes are fairly symmetric around ESPN's number."
    elif pos == "K":
        txt += "Kicker outcomes are close to random around the projection."
    elif pos == "D/ST":
        txt += "D/ST scoring is lumpy: most weeks are modest, with occasional defensive touchdowns."
    return txt


def _props(con, sid: int | None, name: str) -> list[dict]:
    if not sid:
        return []
    psid = con.execute("SELECT MAX(snapshot_id) FROM prop_lines WHERE snapshot_id <= ?", (sid,)).fetchone()[0]
    if not psid:
        return []
    rows = con.execute("SELECT market, outcome, point, price FROM prop_lines WHERE snapshot_id=? AND player_name=?",
                       (psid, name)).fetchall()
    by: dict[tuple, list] = {}
    for m, o, pt, pr in rows:
        by.setdefault((m, o, pt), []).append(pr)

    def implied(price):
        return 100 / (price + 100) if price > 0 else -price / (-price + 100)
    out = []
    for (m, o, pt), prices in sorted(by.items()):
        out.append({"market": m.replace("player_", "").replace("_", " "), "outcome": o, "line": pt,
                    "implied": float(np.median([implied(x) for x in prices])), "books": len(prices)})
    return out


# --------------------------------------------------------------------------- report

def _neutral(df: pl.DataFrame, cols: dict) -> pl.DataFrame:
    return df.with_columns([pl.lit(v).cast(df.schema[c]).alias(c) for c, v in cols.items() if c in df.columns])


def _totals(con, sid: int | None, week: int) -> dict:
    """team -> game total. Live: archived odds; replay: nflverse closing total."""
    if sid is None:
        from ..research.nflverse import load
        s = load("schedules", [SEASON]).filter(pl.col("week") == week)
        return {t: r["total_line"] for r in s.iter_rows(named=True) for t in (r["home_team"], r["away_team"])}
    osid = con.execute("SELECT MAX(snapshot_id) FROM odds_lines WHERE snapshot_id <= ?", (sid,)).fetchone()[0]
    if not osid:
        return {}
    by: dict = {}
    for h, a, ct, pt in con.execute("SELECT home_team, away_team, commence_time, point FROM odds_lines WHERE snapshot_id=?"
                                    " AND market='totals' AND outcome='Over'", (osid,)):
        by.setdefault((TEAM_NAMES.get(h), TEAM_NAMES.get(a), ct[:10]), []).append(pt)
    return {(t, d): float(np.median(v)) for (h, a, d), v in by.items() for t in (h, a)}


def _line_who(source: str, sid: int | None, week: int) -> dict:
    con = _con()
    if source == "live":
        isid = con.execute("SELECT MAX(snapshot_id) FROM injury_reports WHERE snapshot_id <= ?", (sid,)).fetchone()[0]
        rep = pl.DataFrame([dict(r) for r in con.execute(
            "SELECT season, week, gsis_id, report_status FROM injury_reports WHERE snapshot_id=?", (isid,))]) if isid else None
        so = starters_out([SEASON], extra_injuries=rep.with_columns(pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64))
                          if rep is not None and rep.height else None, asof_week=week)
    else:
        so = starters_out([SEASON])
    so = so.filter(pl.col("week") == week)
    out = {}
    for r in so.iter_rows(named=True):
        out[("ol", r["team"])] = r["ol_out_who"] or ""
        out[("db", r["team"])] = r["db_out_who"] or ""
    return out


def _vacated_who(source: str, sid: int | None, week: int) -> dict:
    """team -> (targets share vacated, carries share vacated, who), from what was known before kickoff."""
    from ..research.vacated import vacated
    con = _con()
    if source == "live":
        isid = con.execute("SELECT MAX(snapshot_id) FROM injury_reports WHERE snapshot_id <= ?", (sid,)).fetchone()[0]
        rep = pl.DataFrame([dict(r) for r in con.execute(
            "SELECT season, week, gsis_id, report_status FROM injury_reports WHERE snapshot_id=?", (isid,))]) if isid else None
        v = vacated([SEASON], extra_injuries=rep.with_columns(pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64))
                    if rep is not None and rep.height else None, asof_week=week)
    else:
        v = vacated([SEASON])
    return {r["team"]: (r["vacated_tgt"], r["vacated_car"], r["vacated_who"])
            for r in v.filter(pl.col("week") == week).iter_rows(named=True) if r["vacated_who"]}


def build_report(week: int | None = None, team_id: int | None = None, snapshot_id: int | None = None,
                 overrides: dict | None = None, n: int = 20000, seed: int = 7) -> dict:
    overrides = overrides or {}
    con = _con()
    source, sid, week, taken, pool, ros, statlines = _inputs(week, snapshot_id)
    me_id = team_id or MY_TEAM_ID
    opp_id = ros.filter(pl.col("team_id") == me_id)["opp_team_id"][0]
    counts = _slot_counts(con)

    # what-if: projection nudges
    nudge = {int(k): float(v) for k, v in (overrides.get("proj_delta") or {}).items()}
    if nudge:
        pool = pool.with_columns(pl.col("proj") + pl.col("player_id").replace_strict(nudge, default=0.0))

    df = ros.filter(pl.col("team_id").is_in([me_id, opp_id])).join(pool, on="player_id", how="left").with_columns(
        pl.col("proj").fill_null(0.0),
        (pl.col("kickoff_utc").str.to_datetime(time_zone="UTC") <= pl.lit(taken)).fill_null(False).alias("locked"))
    unavailable = pl.col("report_status").is_in(["Out"]) | (pl.col("slot") == IR) | pl.col("opp").is_null()
    if "played" in df.columns:  # replay: inactive players were known at their game's lock
        unavailable = unavailable | ~pl.col("played").fill_null(False)
    df = df.with_columns(unavailable.fill_null(True).alias("unavailable"))
    df = df.with_columns(pl.col("name").fill_null(pl.lit("?")))
    names_raw = _roster_names(con, source, sid, week)
    df = df.with_columns(pl.struct("player_id", "name").map_elements(
        lambda r: names_raw.get(r["player_id"], (r["name"],))[0] if r["name"] == "?" else r["name"],
        return_dtype=pl.String).alias("name"))

    marg = Marginals.load(MODEL_PATH)
    dfq = df.with_columns(pl.col("pos").fill_null("WR"))
    # live distributions: ESPN's projection + the context's shift (the backtested "ESPN + context" variant)
    q = espn_context_quantiles(marg, dfq)
    taus = list(TAUS)
    i50 = taus.index(0.5)
    proj_arr = dfq["proj"].to_numpy()
    q_ctx_raw = marg.quantiles(dfq)
    mean_ctx = curve_mean(q_ctx_raw)
    # each group's effect on the expectation = what removing just that group changes; they add up
    effects = {g: mean_ctx - curve_mean(marg.quantiles(_neutral(dfq, cols))) for g, cols in GROUPS.items()}
    # the shape before any context, pinned to ESPN's projection (its median vs ESPN = the skew)
    base_med = anchor(marg.quantiles(neutralize(dfq)), proj_arr)[:, i50]
    shares = features(dfq.with_columns([pl.col(c).fill_null(0.0) for c in ["p_rush_td", "p_rec_td", "p_rec", "p_pass_td", "p_rush_yds"]]))

    opp_start = _opponent_starters(df.filter(pl.col("team_id") == opp_id), counts)
    mine = df.filter(pl.col("team_id") == me_id)
    q_of = dict(zip(df["player_id"].to_list(), q))
    dec = decide(mine, opp_start, q_of, counts, C.load(), n=n, seed=seed,
                 force_in=[int(x) for x in overrides.get("force_in", [])],
                 force_out=[int(x) for x in overrides.get("force_out", [])])
    # lineups may hold "EMPTY-..." placeholders for slots nobody available can fill; keep real players only
    best_ids = {p for _, p in dec.lineups[dec.best] if not isinstance(p, str)}
    best_slot = {p: s for s, p in dec.lineups[dec.best] if not isinstance(p, str)}
    mp_ids = {p for _, p in dec.lineups[dec.max_proj] if not isinstance(p, str)}
    mp_slot = {p: s for s, p in dec.lineups[dec.max_proj] if not isinstance(p, str)}
    means = curve_mean(q)  # model's expected points per player
    member = [{p for _, p in lu} for lu in dec.lineups]
    actual_start = set(mine.filter(~pl.col("slot").is_in([BENCH, IR]))["player_id"])

    my_start_rows = mine.filter(pl.col("player_id").is_in(list(best_ids)) | (pl.col("locked") & ~pl.col("slot").is_in([BENCH, IR])))
    my_games = {(r["game"], r["team"], r["pos"], r["name"], r["role"]) for r in my_start_rows.iter_rows(named=True) if r["game"]}
    opp_games = {(r["game"], r["team"], r["pos"], r["name"], r["role"]) for r in opp_start.iter_rows(named=True) if r["game"]}
    cm = C.load()
    ridx = {x: i for i, x in enumerate(cm["roles"])}

    def pair_corr(role_a, team_a, role_b, team_b):
        """Estimated correlation of two players' outcomes (role-pair copula)."""
        if role_a not in ridx or role_b not in ridx:
            return 0.0
        m = cm["same"] if team_a == team_b else cm["opp"]
        if team_a == team_b and role_a == role_b:
            return 0.0
        return float(m[ridx[role_a]][ridx[role_b]])
    totals = _totals(con, sid, week)
    line_who = _line_who(source, sid, week)
    vac_who = _vacated_who(source, sid, week)
    opp_ids = set(opp_start["player_id"])

    def player(i: int, r: dict, side: str) -> dict:
        p = r["player_id"]
        total = totals.get(r["team"]) if source == "replay" else totals.get((r["team"], (r["kickoff_utc"] or "")[:10]))
        out = {
            "id": p, "name": r["name"], "pos": r["pos"] or names_raw.get(p, (None, None))[1], "team": r["team"], "opp": r["opp"],
            "kickoff": r["kickoff_utc"], "slot": SLOT.get(r["slot"], "BE" if r["slot"] == BENCH else "IR"),
            "proj": r["proj"], "model_mean": float(means[i]), "locked": r["locked"], "actual": r["actual"], "final": r.get("final"),
            "unavailable": bool(r["unavailable"]) and not r["locked"], "role": r["role"],
            "status": r["report_status"] or {"INJURY_RESERVE": "IR", "OUT": "Out", "SUSPENSION": "Suspended"}.get(
                (names_raw.get(p) or (None, None, None))[2]), "practice": r["practice_status"], "injury": r["injury"],
            "games_back": r["games_back"], "roof": r["roof"], "wind": r["wind"], "temp": r["temp"],
            "gust": r.get("gust_mph"), "precip_prob": r.get("precip_prob"), "spread": r["exp_margin"], "total": total,
            "ol_out": r["ol_out"], "opp_ol_out": r["opp_ol_out"], "opp_db_out": r["opp_db_out"],
            "quantiles": {f"p{int(t*100)}": float(q[i, taus.index(t)]) for t in TAUS},
            "effects": {g: float(effects[g][i]) for g in GROUPS},
            "td_share": float(shares["f_td_share"][i]), "rec_share": float(shares["f_rec_share"][i]),
            "statline": {STAT_NAMES[k]: round(float(v), 2) for k, v in (statlines.get(p) or {}).items()
                         if k in STAT_NAMES and v}, "nudge": nudge.get(p, 0.0),
        }
        out["explain"] = _explain(out, out["effects"], float(base_med[i] - (r["proj"] or 0)), line_who) if r["pos"] else []
        vt = vac_who.get(r["team"])
        if vt and r["pos"] in ("WR", "TE", "RB"):
            who = "; ".join(x for x in vt[2].split("; ") if not x.startswith(r["name"] + " "))
            if who:
                out["teammates_out"] = who
                grp = "20%+" if vt[0] >= 0.20 else "under 20%" if vt[0] > 0 else "none"
                ev = [_ev("teammate targets vacated", r["pos"], grp)]
                if r["pos"] == "RB":
                    ev.append(_ev("teammate carries vacated", "RB", "40%+" if vt[1] >= 0.40 else "under 40%" if vt[1] > 0 else "none"))
                out["explain"].append({"factor": "teammates", "effect": 0.0, "headline": "Teammates out",
                                       "details": [f"{r['team']} key players out: {who}.",
                                                   f"{vt[0]:.0%} of the team's targets and {vt[1]:.0%} of its carries are vacated.",
                                                   "Not a model input: ESPN's projection already moves for this (e.g. it raises the next players up)."],
                                       "evidence": [e for e in ev if e]})
        out["props"] = _props(con, sid, r["name"])
        if p in (dec.col or {}):
            x = dec.D[:, dec.col[p]]
            out["chance"] = {"beat_proj": float((x > r["proj"]).mean()), "dud": float((x < 5).mean()),
                             "twenty": float((x >= 20).mean())}
        if side == "me":
            out["started_actually"] = p in actual_start
            if not r["locked"] and not out["unavailable"]:
                with_p = [dec.pwin[k] for k, m in enumerate(member) if p in m]
                without = [dec.pwin[k] for k, m in enumerate(member) if p not in m]
                out.update(recommended=p in best_ids, rec_slot=best_slot.get(p), in_max_proj=p in mp_ids,
                           maxproj_slot=mp_slot.get(p),
                           pwin_if_started=float(max(with_p)) if with_p else None,
                           pwin_if_benched=float(max(without)) if without else None)
            out["stacks"] = sorted(({"name": nm, "pos": ps, "team": t, "corr": pair_corr(r["role"], r["team"], rl, t)}
                                    for g, t, ps, nm, rl in my_games
                                    if r["game"] and g == r["game"] and t == r["team"] and nm != r["name"]),
                                   key=lambda x: -abs(x["corr"]))
            out["vs_opponent"] = sorted(({"name": nm, "pos": ps, "team": t, "corr": pair_corr(r["role"], r["team"], rl, t)}
                                         for g, t, ps, nm, rl in opp_games
                                         if r["game"] and g == r["game"] and nm != r["name"]),
                                        key=lambda x: -abs(x["corr"]))
        else:
            out["starting"] = p in opp_ids
        return out

    rows = list(df.iter_rows(named=True))
    me_rows = [player(i, r, "me") for i, r in enumerate(rows) if r["team_id"] == me_id]
    opp_rows = [player(i, r, "opp") for i, r in enumerate(rows) if r["team_id"] == opp_id]

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

    espn_wp = None
    if sid:
        m = con.execute("SELECT * FROM matchups WHERE snapshot_id=? AND matchup_period=? AND ? IN (home_team_id, away_team_id)",
                        (sid, week, me_id)).fetchone()
        if m:
            espn_wp = m["home_win_prob"] if m["home_team_id"] == me_id else m["away_win_prob"]
    names = team_names(con, sid)
    by_id = {r["id"]: dict(r, p10=r["quantiles"]["p10"], p90=r["quantiles"]["p90"],
                           game=next((x["game"] for x in rows if x["player_id"] == r["id"]), None)) for r in me_rows}
    ins, outs = best_ids - mp_ids, mp_ids - best_ids
    reason = _reason(float(dec.pwin[dec.max_proj]), {p for p in ins if p in by_id}, {p for p in outs if p in by_id},
                     by_id, opp_start, df)

    # swaps vs the lineup currently set (live) / actually started (replay)
    cur = mine.filter(~pl.col("slot").is_in([BENCH, IR]) & ~pl.col("locked"))
    cur_ids = set(cur["player_id"])

    def pwin_of(ids):
        cols = [dec.col[p] for p in ids if p in (dec.col or {})]
        tot = dec.fixed_me + dec.D[:, cols].sum(axis=1)
        d = tot - dec.opp_total
        return float((d > 0).mean() + 0.5 * (d == 0).mean())
    rows_by_id = {r["player_id"]: r for r in df.iter_rows(named=True)}

    def brief(p):
        r = rows_by_id[p]
        return {"id": p, "name": r["name"], "pos": r["pos"] or (names_raw.get(p) or (None, None))[1], "team": r["team"], "proj": r["proj"],
                "model_mean": float(means[list(df["player_id"]).index(p)]) if p in dec.col else None,
                "status": r["report_status"] or {"INJURY_RESERVE": "IR", "OUT": "Out", "SUSPENSION": "Suspended"}.get(
                    (names_raw.get(p) or (None, None, None))[2]), "unavailable": bool(r["unavailable"]) and not r["locked"],
                "slot": SLOT.get(r["slot"], "BE")}
    group = {"QB": "QB", "RB": "FLEX", "WR": "FLEX", "TE": "FLEX", "FLEX": "FLEX", "D/ST": "D/ST", "K": "K"}

    def pos_of(p):
        return rows_by_id[p]["pos"] or (names_raw.get(p) or (None, None))[1]
    # outgoing: current starters not in the recommendation, keyed by the slot they occupy now
    swap_out = [(group.get(SLOT.get(rows_by_id[p]["slot"]), "FLEX"), p) for p in cur_ids - best_ids]
    # incoming: recommended players not currently starting, plus empty slots nobody can fill
    swap_in = [(group.get(best_slot.get(p), "FLEX"), p) for p in best_ids - cur_ids]
    swap_in += [(group.get(sl, "FLEX"), None) for sl, p in dec.lineups[dec.best] if isinstance(p, str)]
    pairs = []
    for g in ["QB", "FLEX", "D/ST", "K"]:
        o = sorted([p for gg, p in swap_out if gg == g], key=lambda p: -(rows_by_id[p]["unavailable"] or 0))
        i = sorted([p for gg, p in swap_in if gg == g], key=lambda p: (p is None, -(rows_by_id[p]["proj"] or 0) if p else 0))
        for k in range(max(len(o), len(i))):
            po, pi = (o[k] if k < len(o) else None), (i[k] if k < len(i) else None)
            pairs.append({"out": brief(po) if po else None, "in": brief(pi) if pi else None,
                          "urgent": bool(po and rows_by_id[po]["unavailable"]), "group": g,
                          "note": "no available player at this position" if po and pi is None else None})
    swaps = {"pairs": pairs, "p_win_current": pwin_of(cur_ids), "p_win_recommended": float(dec.pwin[dec.best]),
             "label": "your current lineup" if source == "live" else "the lineup that was started"}

    locked_rows = mine.filter(pl.col("locked") & ~pl.col("slot").is_in([BENCH, IR]))
    proj_of = dict(zip(df["player_id"].to_list(), df["proj"].to_list()))
    mean_of = dict(zip(df["player_id"].to_list(), means.tolist()))
    locked_now = float(locked_rows["actual"].fill_null(0).sum())

    def proj_total(ids):
        return float(locked_now + sum(proj_of.get(p, 0.0) for p in ids))

    def mean_total(ids):
        return float(locked_now + sum(mean_of.get(p, 0.0) for p in ids))
    started_ids = set(mine.filter(~pl.col("slot").is_in([BENCH, IR]) & ~pl.col("locked"))["player_id"])
    lineup_totals = {
        "model": {"espn": proj_total(best_ids), "model": mean_total(best_ids)},
        "max_proj": {"espn": proj_total(mp_ids), "model": mean_total(mp_ids)},
        "started": {"espn": proj_total(started_ids), "model": mean_total(started_ids)},
        "opponent": {"espn": float(opp_start.filter(~pl.col("locked"))["proj"].sum() + opp_start.filter(pl.col("locked"))["actual"].fill_null(0).sum())},
    }

    outcome = None
    if source == "replay":
        fin = dict(zip(mine["player_id"], mine["final"]))
        opp_final = float(opp_start["final"].fill_null(0).sum())
        locked_pts = float(mine.filter(pl.col("locked") & ~pl.col("slot").is_in([BENCH, IR]))["final"].fill_null(0).sum())

        def realized(ids):
            return locked_pts + sum(fin.get(p) or 0.0 for p in ids)
        unlocked_actual = set(mine.filter(~pl.col("slot").is_in([BENCH, IR]) & ~pl.col("locked"))["player_id"])
        outcome = {"opponent": opp_final, "model": realized(best_ids), "max_proj": realized(mp_ids),
                   "actual": realized(unlocked_actual)}
    return {
        "source": source, "snapshot_id": sid, "week": week, "decision_time": taken.isoformat(),
        "freshness": _freshness(con, sid) if sid else {"replay": "ESPN projections + nflverse context as of Sunday 11:45 ET"},
        "reason": reason, "overrides": overrides,
        "me": {"id": me_id, "name": names.get(me_id)}, "opponent": {"id": opp_id, "name": names.get(opp_id)},
        "p_win": float(dec.pwin[dec.best]), "p_win_max_proj": float(dec.pwin[dec.max_proj]), "espn_win_prob": espn_wp,
        "my_mean": float(dec.my_totals[:, dec.best].mean()), "opp_mean": float(dec.opp_total.mean()),
        "my_range": [float(np.percentile(dec.my_totals[:, dec.best], p)) for p in (10, 50, 90)],
        "opp_range": [float(np.percentile(dec.opp_total, p)) for p in (10, 50, 90)],
        "margin_hist": {"edges": edges.tolist(), "counts": hist.tolist(), "n": int(len(margin))},
        "changes": {"start": sorted(info[p] for p in ins if p in info), "bench": sorted(info[p] for p in outs if p in info)},
        "close": bool(len(order) > 1 and dec.pwin[order[0]] - dec.pwin[order[1]] < 0.01),
        "alternatives": [alt(int(k)) for k in order[1:6]],
        "locked_count": int(mine.filter(pl.col("locked") & ~pl.col("slot").is_in([BENCH, IR])).height),
        "players": {"me": me_rows, "opp": opp_rows}, "outcome": outcome, "lineup_totals": lineup_totals, "draws": n,
        "swaps": swaps,
        "teams": [{"id": k, "name": v} for k, v in sorted(names.items(), key=lambda kv: kv[1].lower())],
    }


def _roster_names(con, source, sid, week) -> dict[int, tuple]:
    """Names/positions for rostered players without projections (byes, IR), from ESPN's roster data."""
    pos_map = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "D/ST"}
    if source == "live":
        rows = con.execute("SELECT * FROM raw_responses WHERE snapshot_id=? AND kind='league' AND status_code=200", (sid,)).fetchall()
    else:
        rows = [r for r in con.execute("SELECT * FROM raw_responses WHERE kind='prev_week' AND status_code=200")
                if json.loads(r["params_json"]).get("scoringPeriodId") == week]
    out = {}
    for row in rows[-1:]:
        for t in load_raw(row)["teams"]:
            for e in t["roster"]["entries"]:
                p = e["playerPoolEntry"]["player"]
                out[p["id"]] = (p["fullName"], pos_map.get(p["defaultPositionId"]), p.get("injuryStatus"))
    return out


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


# --------------------------------------------------------------------------- week context & season tracking

def week_context(week: int | None = None, snapshot_id: int | None = None) -> dict:
    """The raw context for a whole week: games, injury report, OL/DB starters out."""
    from ..research.nflverse import load
    con = _con()
    source, sid, week, taken, pool, ros, _ = _inputs(week, snapshot_id)
    sch = load("schedules", [SEASON]).filter(pl.col("week") == week)
    games = []
    gc = {}
    if sid:
        csid = con.execute("SELECT MAX(snapshot_id) FROM game_context WHERE snapshot_id <= ?", (sid,)).fetchone()[0]
        gc = {(r["home"], r["away"]): dict(r) for r in con.execute("SELECT * FROM game_context WHERE snapshot_id=?", (csid,))}
    margins = {r["team"]: r["exp_margin"] for r in pool.select("team", "exp_margin").unique("team").iter_rows(named=True)}
    totals = _totals(con, sid, week)
    for r in sch.sort("gameday", "gametime").iter_rows(named=True):
        g = gc.get((r["home_team"], r["away_team"]), {})
        hm = margins.get(r["home_team"])
        total = totals.get(r["home_team"]) if source == "replay" else next(
            (v for (t, d), v in totals.items() if t == r["home_team"]), None)
        games.append({
            "away": r["away_team"], "home": r["home_team"], "kickoff": g.get("kickoff_utc") or f"{r['gameday']} {r['gametime']} ET",
            "stadium": g.get("stadium") or r["stadium"], "roof": g.get("roof") or r["roof"],
            "temp": g.get("temp_f") if g else r["temp"], "wind": g.get("wind_mph") if g else r["wind"],
            "gust": g.get("gust_mph"), "precip_prob": g.get("precip_prob"),
            "weather_kind": "forecast" if g else "measured at game time (nflverse)",
            "home_spread": -hm if hm is not None else None, "total": total,
            "score": f"{r['away_score']}–{r['home_score']}" if r["home_score"] is not None and source == "replay" else None})
    # injury report
    if sid:
        isid = con.execute("SELECT MAX(snapshot_id) FROM injury_reports WHERE snapshot_id <= ?", (sid,)).fetchone()[0]
        inj = [dict(r) for r in con.execute("SELECT team, full_name, position, report_status, practice_status, injury"
                                            " FROM injury_reports WHERE snapshot_id=?", (isid,))] if isid else []
        espn = [dict(name=r["name"], team=r["team"], pos=r["pos"], status=r["report_status"])
                for r in pool.filter(pl.col("report_status") != "none").select("name", "team", "pos", "report_status").iter_rows(named=True)]
    else:
        i = load("injuries", [SEASON]).filter(pl.col("week") == week)
        inj = [dict(team=r["team"], full_name=r["full_name"], position=r["position"], report_status=r["report_status"],
                    practice_status=r["practice_status"], injury=r["report_primary_injury"]) for r in i.iter_rows(named=True)]
        espn = []
    lw = _line_who(source, sid, week)
    lines = sorted([{"team": t, "unit": u.upper(), "who": who} for (u, t), who in lw.items() if who], key=lambda x: (x["team"], x["unit"]))
    return {"source": source, "week": week, "snapshot_id": sid, "games": games, "injuries": inj, "espn_status": espn,
            "lines": lines}


def season_tracking(team_id: int | None = None) -> dict:
    """Per week: what the model would have started vs. the highest projection vs. what the team started."""
    p = DATA_DIR / "backtest_league_2026.parquet"
    me = team_id or MY_TEAM_ID
    names = team_names()
    rows = []
    if p.exists():
        r = pl.read_parquet(p).filter(pl.col("team_id") == me).sort("week")
        for x in r.iter_rows(named=True):
            rows.append({"week": x["week"], "opponent": names.get(x["opp_team_id"]), "source": "replay",
                         "model_points": x["model_points"], "maxproj_points": x["maxproj_points"],
                         "actual_points": x["actual_points"], "opp_points": x["opp_points"],
                         "model_pwin": x["model_pwin"], "actual_pwin": x["actual_pwin"], "differs": x["model_differs"],
                         "won": x["actual_win"] == 1, "model_would_win": x["model_win"] == 1})
    return {"team": {"id": me, "name": names.get(me)}, "weeks": rows}
