"""Pick the lineup that maximizes P(win this week's matchup).

Uses one archived snapshot: player inputs and ranges (live.py + marginals),
both rosters and current lineups, and which games have kicked off. Players
whose game has started are locked in their slot at their current points (late
swap). The opponent keeps their current starters (an Out starter is replaced by
their best-projected eligible bench player). All players are drawn jointly with
the game-level copula, and every legal lineup is scored on the same draws.
"""
from __future__ import annotations

import itertools
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import polars as pl

from ..config import DB_PATH, MY_TEAM_ID
from ..db import load_raw
from . import correlation as C
from .live import build_inputs
from .marginals import MODEL_PATH, TAUS, Marginals
from .simulate import draw

SLOT = {0: "QB", 2: "RB", 4: "WR", 6: "TE", 16: "D/ST", 17: "K", 23: "FLEX"}
BENCH, IR = 20, 21
FLEX_OK = {"RB", "WR", "TE"}


@dataclass
class Recommendation:
    snapshot_id: int
    week: int
    opponent_team_id: int
    best: list[dict]                 # [{slot, name, pos, proj, p10, p50, p90, locked}]
    p_win: float
    p_win_max_proj: float
    max_proj_lineup: list[dict]
    changes: list[str]
    close: bool
    reason: str
    my_mean: float
    opp_mean: float
    alternatives: list[tuple[float, list[str]]] = field(default_factory=list)


def _slot_counts(con) -> dict[str, int]:
    row = con.execute("SELECT * FROM raw_responses WHERE kind='settings' AND status_code=200 ORDER BY id DESC LIMIT 1").fetchone()
    counts = load_raw(row)["settings"]["rosterSettings"]["lineupSlotCounts"]
    return {SLOT[int(k)]: v for k, v in counts.items() if int(k) in SLOT and v}


def _rosters(con, sid: int, week: int):
    m = con.execute("SELECT home_team_id, away_team_id FROM matchups WHERE snapshot_id=? AND matchup_period=?"
                    " AND ? IN (home_team_id, away_team_id)", (sid, week, MY_TEAM_ID)).fetchone()
    opp = m[1] if m[0] == MY_TEAM_ID else m[0]
    rows = con.execute("SELECT team_id, player_id, lineup_slot_id, actual_points FROM lineups"
                       " WHERE snapshot_id=? AND scoring_period=? AND team_id IN (?, ?)", (sid, week, MY_TEAM_ID, opp)).fetchall()
    df = pl.DataFrame([dict(team_id=r[0], player_id=r[1], slot=r[2], actual=r[3]) for r in rows])
    return opp, df


def recommend(snapshot_id: int | None = None, n: int = 20000, seed: int = 7) -> Recommendation:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    pool = build_inputs(snapshot_id)
    sid = int(pool["snapshot_id"][0])
    week = int(pool["week"][0])
    taken = datetime.fromisoformat(con.execute("SELECT started_at FROM snapshots WHERE id=?", (sid,)).fetchone()[0])
    counts = _slot_counts(con)
    opp_id, ros = _rosters(con, sid, week)

    pool = C.assign_roles(pool)  # roles from the whole player pool, by NFL team
    df = ros.join(pool, on="player_id", how="left").with_columns(
        pl.col("proj").fill_null(0.0), pl.col("name").fill_null("?"),
        (pl.col("kickoff_utc").str.to_datetime(time_zone="UTC") <= pl.lit(taken)).fill_null(False).alias("locked"),
        (pl.col("report_status").is_in(["Out"]) | (pl.col("slot") == IR) | pl.col("opp").is_null()).fill_null(True)
        .alias("unavailable"))
    q = Marginals.load(MODEL_PATH).quantiles(df.with_columns(pl.col("pos").fill_null("WR")))
    taus = list(TAUS)
    df = df.with_columns(*[pl.Series(f"p{int(t*100)}", q[:, taus.index(t)]) for t in (0.1, 0.5, 0.9)])

    # opponent starters: current lineup, Out/bye starters replaced by best eligible bench
    opp = df.filter(pl.col("team_id") == opp_id)
    opp_start = _opponent_starters(opp, counts)

    mine = df.filter(pl.col("team_id") == MY_TEAM_ID)
    my_locked = mine.filter(pl.col("locked") & (pl.col("slot") != BENCH) & (pl.col("slot") != IR))
    free = mine.filter(~pl.col("locked") & ~pl.col("unavailable"))
    open_slots = dict(counts)
    for s in my_locked["slot"]:
        open_slots[SLOT[s]] -= 1
    lineups = _enumerate(free, open_slots)

    # simulate every unlocked player on either roster who could score for us or them
    sim_ids = sorted(set(free["player_id"]) | set(opp_start.filter(~pl.col("locked"))["player_id"]))
    S = df.filter(pl.col("player_id").is_in(sim_ids)).unique("player_id").sort("player_id")
    col = {p: i for i, p in enumerate(S["player_id"])}
    teams = S["team"].to_list()
    sides = [0 if t == min(t, o) else 1 for t, o in zip(teams, S["opp"].to_list())]
    D = draw(q[[df["player_id"].to_list().index(p) for p in S["player_id"]]], S["game"].to_list(),
             S["role"].to_list(), sides, C.load(), n=n, rng=np.random.default_rng(seed))

    fixed_me = float(my_locked["actual"].fill_null(0).sum())
    opp_total = float(opp_start.filter(pl.col("locked"))["actual"].fill_null(0).sum()) + \
        D[:, [col[p] for p in opp_start.filter(~pl.col("locked"))["player_id"]]].sum(axis=1)
    L = np.zeros((len(lineups), len(col)))
    for k, lu in enumerate(lineups):
        L[k, [col[p] for _, p in lu]] = 1.0
    my_totals, diff, pwin = score_lineups(D, L, fixed_me, opp_total)

    proj = dict(zip(free["player_id"], free["proj"]))
    mp = int(np.argmax([sum(proj[p] for _, p in lu) for lu in lineups]))
    best = int(np.argmax(pwin))
    order = np.argsort(-pwin)

    info = {r["player_id"]: r for r in df.iter_rows(named=True)}

    def describe(lu):
        rows = [dict(slot=SLOT[r["slot"]], name=r["name"], pos=r["pos"], proj=r["proj"], p10=r["p10"], p50=r["p50"],
                     p90=r["p90"], locked=True, actual=r["actual"]) for r in my_locked.iter_rows(named=True)]
        rows += [dict(slot=s, name=info[p]["name"], pos=info[p]["pos"], proj=info[p]["proj"], p10=info[p]["p10"],
                      p50=info[p]["p50"], p90=info[p]["p90"], locked=False) for s, p in lu]
        return rows

    b_ids, m_ids = {p for _, p in lineups[best]}, {p for _, p in lineups[mp]}
    ins, outs = b_ids - m_ids, m_ids - b_ids
    changes = [f"start {info[p]['name']} ({info[p]['pos']}, proj {info[p]['proj']:.1f})" for p in ins] + \
              [f"bench {info[p]['name']} ({info[p]['pos']}, proj {info[p]['proj']:.1f})" for p in outs]
    # close call: the best lineup beats its runner-up by < 1 pt of P(win) or within Monte Carlo noise
    runner = int(order[1]) if len(order) > 1 else best
    d = (diff[:, best] > 0).astype(float) - (diff[:, runner] > 0).astype(float)
    se = d.std() / np.sqrt(n)
    close = (pwin[best] - pwin[runner]) < max(0.01, 2 * se)
    reason = _reason(pwin[mp], ins, outs, info, opp_start, df)
    return Recommendation(
        snapshot_id=sid, week=week, opponent_team_id=opp_id, best=describe(lineups[best]), p_win=float(pwin[best]),
        p_win_max_proj=float(pwin[mp]), max_proj_lineup=describe(lineups[mp]), changes=changes, close=bool(close),
        reason=reason, my_mean=float(my_totals[:, best].mean()), opp_mean=float(opp_total.mean()),
        alternatives=[(float(pwin[k]), [info[p]["name"] for _, p in lineups[k]]) for k in order[1:4]])


def score_lineups(D: np.ndarray, L: np.ndarray, fixed_me: float, opp_total: np.ndarray):
    """D: (draws, players) simulated points; L: (lineups, players) 0/1 starters.
    Returns my totals (draws, lineups), margin, and P(win) per lineup (ties count half)."""
    my_totals = fixed_me + D @ L.T
    diff = my_totals - np.asarray(opp_total)[:, None]
    return my_totals, diff, (diff > 0).mean(axis=0) + 0.5 * (diff == 0).mean(axis=0)


def _opponent_starters(opp: pl.DataFrame, counts: dict[str, int]) -> pl.DataFrame:
    start = opp.filter(~pl.col("slot").is_in([BENCH, IR]))
    bench = opp.filter((pl.col("slot") == BENCH) & ~pl.col("unavailable") & ~pl.col("locked")).sort("proj", descending=True)
    keep, used = [], set()
    for r in start.iter_rows(named=True):
        if r["unavailable"] and not r["locked"]:
            need = SLOT[r["slot"]]
            ok = FLEX_OK if need == "FLEX" else {need}
            sub = next((b for b in bench.iter_rows(named=True) if b["pos"] in ok and b["player_id"] not in used), None)
            if sub:
                used.add(sub["player_id"])
                keep.append(sub["player_id"])
            continue
        keep.append(r["player_id"])
    return opp.filter(pl.col("player_id").is_in(keep))


def _enumerate(free: pl.DataFrame, open_slots: dict[str, int]) -> list[list[tuple[str, int]]]:
    by_pos = {p: free.filter(pl.col("pos") == p)["player_id"].to_list() for p in ["QB", "RB", "WR", "TE", "K", "D/ST"]}
    singles = {}
    for pos in ["QB", "K", "D/ST"]:
        k = open_slots.get(pos, 0)
        singles[pos] = [list(c) for c in itertools.combinations(by_pos[pos], k)] or [[]]
    out = []
    for rb in itertools.combinations(by_pos["RB"], open_slots.get("RB", 0)):
        for wr in itertools.combinations(by_pos["WR"], open_slots.get("WR", 0)):
            for te in itertools.combinations(by_pos["TE"], open_slots.get("TE", 0)):
                used = set(rb) | set(wr) | set(te)
                flex_pool = [p for p in by_pos["RB"] + by_pos["WR"] + by_pos["TE"] if p not in used]
                for fx in itertools.combinations(flex_pool, open_slots.get("FLEX", 0)):
                    core = [("RB", p) for p in rb] + [("WR", p) for p in wr] + [("TE", p) for p in te] + \
                           [("FLEX", p) for p in fx]
                    for qb in singles["QB"]:
                        for k in singles["K"]:
                            for d in singles["D/ST"]:
                                out.append(core + [("QB", p) for p in qb] + [("K", p) for p in k] + [("D/ST", p) for p in d])
    # the same players with WR/FLEX labels swapped is the same lineup
    seen, unique = set(), []
    for lu in out:
        key = frozenset(p for _, p in lu)
        if key not in seen:
            seen.add(key)
            unique.append(lu)
    return unique


def _reason(p_base: float, ins, outs, info, opp_start, df) -> str:
    if not ins:
        return "The highest-projection lineup is already the best bet."
    status = "favorite" if p_base > 0.55 else "underdog" if p_base < 0.45 else "toss-up"
    parts = [f"You're a {status} ({p_base:.0%} with the highest-projection lineup)."]
    opp_games = set(opp_start["game"].drop_nulls())
    for p_in, p_out in zip(sorted(ins), sorted(outs)):
        a, b = info[p_in], info[p_out]
        wa, wb = a["p90"] - a["p10"], b["p90"] - b["p10"]
        bits = []
        if status == "favorite" and wa < wb:
            bits.append(f"{a['name']} has the safer range ({a['p10']:.1f}-{a['p90']:.1f} vs {b['p10']:.1f}-{b['p90']:.1f})")
        elif status == "underdog" and a["p90"] > b["p90"]:
            bits.append(f"{a['name']} has the higher ceiling ({a['p90']:.1f} vs {b['p90']:.1f})")
        if a.get("game") in opp_games and b.get("game") not in opp_games:
            bits.append(f"{a['name']} plays in the same game as your opponent's starters (hedge)")
        if not bits:
            bits.append(f"{a['name']} over {b['name']} fits better with the rest of your lineup")
        parts.append("; ".join(bits) + ".")
    return " ".join(parts)


def to_json(r: Recommendation) -> str:
    return json.dumps(r.__dict__, default=float, indent=1)


def format_text(r: Recommendation) -> str:
    order = ["QB", "RB", "WR", "TE", "FLEX", "D/ST", "K"]
    lines = [f"Week {r.week} (snapshot {r.snapshot_id}) vs team {r.opponent_team_id}",
             f"P(win): {r.p_win:.1%} with the recommended lineup, {r.p_win_max_proj:.1%} with the highest-projection lineup"
             f" (gain {100 * (r.p_win - r.p_win_max_proj):+.1f} pts)",
             "Decision: " + ("close call -- the runner-up lineup is within 1 pt of P(win)" if r.close
                             else "clear -- the best lineup beats the runner-up by more than 1 pt of P(win)"),
             f"Expected score: you {r.my_mean:.1f}, opponent {r.opp_mean:.1f}", ""]
    for p in sorted(r.best, key=lambda x: order.index(x["slot"])):
        tag = f"LOCKED {p['actual']:.1f} pts" if p["locked"] else f"proj {p['proj']:5.1f}  range {p['p10']:4.1f} / {p['p50']:4.1f} / {p['p90']:4.1f}"
        lines.append(f"  {p['slot']:5s} {p['name']:24s} {tag}")
    lines += ["", "Changes vs highest projection: " + ("; ".join(r.changes) if r.changes else "none"),
              "Why: " + r.reason]
    if r.alternatives:
        best_names = {p["name"] for p in r.best if not p["locked"]}
        alts = []
        for pw, names in r.alternatives:
            swap_in = sorted(set(names) - best_names)
            swap_out = sorted(best_names - set(names))
            alts.append(f"{pw:.1%} ({', '.join(swap_in)} for {', '.join(swap_out)})")
        lines.append("Runner-up lineups: " + " | ".join(alts))
    return "\n".join(lines)
