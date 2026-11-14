"""A shareable, league-free ranking of every player in this week's pool.

Same distributions and explanations as the recommendation page ("ESPN + context"),
for every projected player rather than one matchup. Nothing league-specific goes in:
no rosters, team names or IDs; only the scoring rules, stated in plain words.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from ..config import DATA_DIR
from ..research.scoring import league_rules
from .marginals import MODEL_PATH, TAUS, Marginals, anchor, curve_mean, espn_context_quantiles, features, neutralize
from .report import GROUPS, STAT_NAMES, _con, _explain, _inputs, _line_who, _neutral, _props, _totals, _vacated_who, _ev

TEMPLATE = Path(__file__).resolve().parent.parent / "web" / "static" / "rankings.html"
OUT_DIR = DATA_DIR / "share"
POSITIONS = ("QB", "RB", "WR", "TE", "K", "D/ST")
MIN_PROJ = 1.0

# stat id -> plain-English label, for the scoring panel
RULE_LABELS = {
    3: "per passing yard", 4: "passing TD", 19: "passing 2-pt conversion", 20: "interception thrown",
    24: "per rushing yard", 25: "rushing TD", 26: "rushing 2-pt conversion",
    42: "per receiving yard", 43: "receiving TD", 44: "receiving 2-pt conversion", 53: "reception",
    72: "fumble lost", 63: "fumble recovered for TD", 101: "kickoff return TD", 102: "punt return TD",
    80: "FG made 0–39 yds", 77: "FG made 40–49 yds", 198: "FG made 50–59 yds", 201: "FG made 60+ yds",
    85: "FG missed (incl. blocked)", 86: "extra point made",
    99: "sack", 95: "interception", 96: "fumble recovered", 93: "blocked punt/FG returned for TD",
    97: "blocked kick", 98: "fumble return TD", 103: "interception return TD", 104: "fumble return TD",
    206: "2-pt return", 209: "1-pt safety",
}
PA_LABELS = {89: "0", 90: "1–6", 91: "7–13", 92: "14–17", 123: "28–34", 124: "35–45", 125: "46+"}
YA_LABELS = {128: "<100", 129: "100–199", 130: "200–299", 132: "350–399", 133: "400–449", 134: "450–499",
             135: "500–549", 136: "550+"}


def _scoring() -> dict:
    rules = league_rules()
    off, dst = [], []
    for sid, (p, d) in sorted(rules.items()):
        if sid in RULE_LABELS and p:
            off.append({"what": RULE_LABELS[sid], "pts": p})
        elif sid in RULE_LABELS and d:
            dst.append({"what": RULE_LABELS[sid], "pts": d})
    pa = [{"what": f"{lab} pts allowed", "pts": rules[s][1]} for s, lab in PA_LABELS.items() if s in rules]
    ya = [{"what": f"{lab} yds allowed", "pts": rules[s][1]} for s, lab in YA_LABELS.items() if s in rules]
    p = {s: v[0] for s, v in rules.items()}
    summary = (f"{'Full' if p.get(53) == 1 else 'Half' if p.get(53) == 0.5 else 'No'} PPR · "
               f"{p.get(4, 0):g}-pt passing TD · 1 pt per {1 / p[3]:.0f} passing yds · "
               f"1 pt per {1 / p[24]:.0f} rushing/receiving yds · INT {p.get(20, 0):g} · fumble lost {p.get(72, 0):g}")
    return {"summary": summary, "offense": off, "dst": dst, "points_allowed": pa, "yards_allowed": ya,
            "note": "Tiers not listed (e.g. 18–27 points allowed, 300–349 yards allowed) score 0."}


def _chances(q: np.ndarray, proj: float) -> dict:
    taus = np.asarray(TAUS)
    cdf = lambda x: float(np.interp(x, q, taus, left=0.0, right=1.0))  # noqa: E731
    return {"beat_proj": 1 - cdf(proj), "dud": cdf(5.0), "twenty": 1 - cdf(20.0)}


def build(snapshot_id: int | None = None) -> dict:
    con = _con()
    source, sid, week, taken, pool, _, statlines = _inputs(None, snapshot_id)
    df = pool.filter(pl.col("pos").is_in(POSITIONS) & (pl.col("proj") >= MIN_PROJ) & pl.col("opp").is_not_null()
                     & ~pl.col("report_status").fill_null("none").is_in(["Out"]))
    df = df.with_columns(
        (pl.col("kickoff_utc").str.to_datetime(time_zone="UTC") <= pl.lit(taken)).fill_null(False).alias("locked"))

    marg = Marginals.load(MODEL_PATH)
    q = espn_context_quantiles(marg, df)
    taus = list(TAUS)
    i50 = taus.index(0.5)
    proj = df["proj"].to_numpy()
    mean_ctx = curve_mean(marg.quantiles(df))
    effects = {g: mean_ctx - curve_mean(marg.quantiles(_neutral(df, cols))) for g, cols in GROUPS.items()}
    base_med = anchor(marg.quantiles(neutralize(df)), proj)[:, i50]
    shares = features(df.with_columns([pl.col(c).fill_null(0.0) for c in
                                       ["p_rush_td", "p_rec_td", "p_rec", "p_pass_td", "p_rush_yds"]]))
    means = curve_mean(q)
    totals = _totals(con, sid, week)
    line_who = _line_who(source, sid, week)
    vac_who = _vacated_who(source, sid, week)

    players = []
    for i, r in enumerate(df.iter_rows(named=True)):
        p = {
            "id": r["player_id"], "name": r["name"], "pos": r["pos"], "team": r["team"], "opp": r["opp"],
            "kickoff": r["kickoff_utc"], "locked": r["locked"], "proj": round(float(r["proj"]), 2),
            "mean": round(float(means[i]), 2), "status": r["report_status"], "practice": r["practice_status"],
            "injury": r["injury"], "games_back": r["games_back"], "roof": r["roof"], "wind": r["wind"],
            "temp": r["temp"], "gust": r.get("gust_mph"), "precip_prob": r.get("precip_prob"),
            "spread": r["exp_margin"], "total": totals.get((r["team"], (r["kickoff_utc"] or "")[:10])),
            "ol_out": r["ol_out"], "opp_ol_out": r["opp_ol_out"], "opp_db_out": r["opp_db_out"],
            "td_share": float(shares["f_td_share"][i]), "rec_share": float(shares["f_rec_share"][i]),
            "q": [round(float(x), 1) for x in q[i]],
            "effects": {g: round(float(effects[g][i]), 2) for g in GROUPS},
            "chance": _chances(q[i], float(r["proj"])),
            "statline": {STAT_NAMES[k]: round(float(v), 1) for k, v in (statlines.get(r["player_id"]) or {}).items()
                         if k in STAT_NAMES and v},
        }
        p["explain"] = _explain(p, p["effects"], float(base_med[i] - r["proj"]), line_who)
        vt = vac_who.get(r["team"])
        if vt and r["pos"] in ("WR", "TE", "RB"):
            who = "; ".join(x for x in vt[2].split("; ") if not x.startswith(r["name"] + " "))
            if who:
                grp = "20%+" if vt[0] >= 0.20 else "under 20%" if vt[0] > 0 else "none"
                p["explain"].append({"factor": "teammates", "effect": 0.0, "headline": "Teammates out",
                                     "details": [f"{r['team']} key players out: {who}.",
                                                 f"{vt[0]:.0%} of the team's targets and {vt[1]:.0%} of its carries are vacated.",
                                                 "Not a model input: ESPN's projection already moves for this."],
                                     "evidence": [e for e in [_ev("teammate targets vacated", r["pos"], grp)] if e]})
        p["props"] = [x for x in _props(con, sid, r["name"]) if x["outcome"] in ("Over", "Yes")]
        for k in ("roof", "wind", "temp", "gust", "precip_prob", "ol_out", "opp_ol_out", "opp_db_out",
                  "td_share", "rec_share", "games_back"):
            p.pop(k)
        players.append(p)
    players.sort(key=lambda x: -x["mean"])
    fetched = con.execute("SELECT started_at FROM snapshots WHERE id=?", (sid,)).fetchone()[0] if sid else None
    return {"week": week, "season": int(pool["season"][0]), "as_of": fetched, "taus": taus,
            "scoring": _scoring(), "players": players}


def write(snapshot_id: int | None = None) -> Path:
    data = build(snapshot_id)
    html = TEMPLATE.read_text().replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"rankings-{data['season']}-w{data['week']:02d}.html"
    out.write_text(html)
    return out
