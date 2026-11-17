"""A shareable, league-free ranking of every player in this week's pool.

Same distributions and explanations as the recommendation page ("ESPN + context"),
for every projected player rather than one matchup. Nothing league-specific goes in:
no rosters, team names or IDs; only the scoring rules, stated in plain words.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from ..config import DATA_DIR
from ..research.scoring import league_rules
from .marginals import MODEL_PATH, TAUS, Marginals, anchor, curve_mean, espn_context_quantiles, features, neutralize
from .report import GROUPS, STAT_NAMES, _con, _decision_snapshot, _final_weeks, _live_weeks, _explain, _inputs, _line_who, _neutral, _props, _totals, _vacated_who, _ev

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


# prop market -> ESPN stat id it lines up with
MARKET_STAT = {"player_receptions": "53", "player_reception_yds": "42", "player_rush_yds": "24", "player_pass_yds": "3"}


def _book_median(lines: list[tuple[float, float]]) -> float:
    """The books' median from (line, no-vig chance of the over) pairs: where the over is a coin flip."""
    lines = sorted(lines)
    for (a, pa), (b, pb) in zip(lines, lines[1:]):
        if pa >= 0.5 >= pb and pa > pb:
            return a + (b - a) * (pa - 0.5) / (pa - pb)
    return min(lines, key=lambda x: abs(x[1] - 0.5))[0]


def _books(props: list[dict], statline: dict, pos: str) -> list[dict]:
    """The books' number for each stat ESPN projects, from the player props.

    Yardage and receptions: the books' median (where the no-vig over is a coin flip). Touchdowns:
    expected rushing + receiving TDs (a QB's own runs only) from the anytime-TD price.
    """
    by: dict[str, dict] = {}
    for x in props:
        by.setdefault(x["market"], {}).setdefault(x["line"], {})[x["outcome"]] = x["implied"]
    out = []
    for m, sid in MARKET_STAT.items():
        lines = [(ln, o["Over"] / (o["Over"] + o["Under"])) for ln, o in by.get(m.replace("player_", "").replace("_", " "), {}).items()
                 if ln is not None and "Over" in o and "Under" in o]
        if lines:
            out.append({"stat": STAT_NAMES[sid], "sid": int(sid), "espn": float(statline.get(sid) or 0.0),
                        "book": _book_median(lines)})
    td = by.get("anytime td", {}).get(None, {}).get("Yes")
    if td is not None:
        espn = sum(float(statline.get(k) or 0.0) for k in (("25",) if pos == "QB" else ("25", "43")))
        out.append({"stat": "TDs", "sid": None, "espn": espn, "book": -math.log(1 - min(td, 0.95)), "chance": td})
    return out


def _books_summary(players: list[dict], rules: dict) -> dict:
    """Put the books on ESPN's scale, then turn each player's differences into points.

    Raw props sit systematically off ESPN: a line is a median, below the average ESPN projects
    because yardage is skewed, and anytime-TD prices carry the books' margin. Each stat is divided
    by its slate-wide ratio (books' total over ESPN's), so what's left is where the books are higher or lower on a
    player than they usually are. Information only, not a model input.
    """
    ratio = {}
    for k in {x["stat"] for p in players for x in p["books"]}:
        xs = [x for p in players for x in p["books"] if x["stat"] == k and x["espn"] > 0]
        if len(xs) >= 10:
            ratio[k] = sum(x["book"] for x in xs) / sum(x["espn"] for x in xs)
    for p in players:
        stats, delta = [], 0.0
        for x in p["books"]:
            if x["stat"] not in ratio:
                continue
            adj = x["book"] / ratio[x["stat"]]
            delta += (adj - x["espn"]) * (6 if x["sid"] is None else rules.get(x["sid"], (0, 0))[0])
            stats.append({"stat": x["stat"], "espn": round(x["espn"], 2), "book": round(adj, 2), "raw": round(x["book"], 2),
                          **({"chance": round(x["chance"], 3)} if "chance" in x else {})})
        p["books"] = {"points": round(p["proj"] + delta, 2), "stats": stats, "as_of": p.pop("_props_at")} if stats else None
        p.pop("_props_at", None)
    return {k: round(v, 3) for k, v in sorted(ratio.items())}


def _chances(q: np.ndarray, proj: float) -> dict:
    taus = np.asarray(TAUS)
    cdf = lambda x: float(np.interp(x, q, taus, left=0.0, right=1.0))  # noqa: E731
    return {"beat_proj": 1 - cdf(proj), "dud": cdf(5.0), "twenty": 1 - cdf(20.0)}


def build(snapshot_id: int | None = None, week: int | None = None) -> dict:
    """Rankings at a snapshot, or for a week: a finished live week as of its Sunday 11:45 snapshot,
    W1-9 replayed from ESPN's projection history, the current week from its latest snapshot."""
    con = _con()
    if snapshot_id is None and week is not None and week in _live_weeks(con) and week in _final_weeks(con):
        snapshot_id = _decision_snapshot(con, week)
    source, sid, week, taken, pool, _, statlines = _inputs(None if snapshot_id else week, snapshot_id)
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
    rules = league_rules()

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
        props = _props(con, sid, r["name"])
        p["books"] = _books(props, statlines.get(r["player_id"]) or {}, r["pos"])
        p["_props_at"] = props[0]["as_of"] if props else None
        for k in ("roof", "wind", "temp", "gust", "precip_prob", "ol_out", "opp_ol_out", "opp_db_out",
                  "td_share", "rec_share", "games_back"):
            p.pop(k)
        players.append(p)
    players.sort(key=lambda x: -x["mean"])
    books_scale = _books_summary(players, rules)
    return {"week": week, "season": int(pool["season"][0]), "as_of": taken.isoformat(), "replay": source == "replay", "taus": taus,
            "scoring": _scoring(), "books_scale": books_scale, "players": players}


def render(data: dict) -> str:
    return TEMPLATE.read_text().replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))


def write(snapshot_id: int | None = None, week: int | None = None) -> Path:
    data = build(snapshot_id, week)
    html = render(data)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"rankings-{data['season']}-w{data['week']:02d}.html"
    out.write_text(html)
    return out
