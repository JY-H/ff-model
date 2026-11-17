"""Post-week review: was a result role or luck?

For each player in a finished week (see finished_weeks: live weeks join once Tuesday's finals land):
  proj   ESPN's projection
  xfp    expected points from the opportunities he actually got (targets, carries, attempts,
         and where on the field they happened; nflverse ff_opportunity, this league's scoring)
  actual ESPN's final points
  role = xfp - proj     ESPN's miss on how much he'd be used
  luck = actual - xfp   what he did with that usage (efficiency, TDs, big plays)
So actual = proj + role + luck. K and D/ST have no opportunity model: their whole miss is "K/DST".

Information only: the 2019-2025 screen found neither part predicts next week better than ESPN.
"""
from __future__ import annotations

from functools import lru_cache

import polars as pl

from ..config import DATA_DIR, LEAGUE_ID, MY_TEAM_ID, SEASON
from ..research.nflverse import NFLVERSE_DIR, load
from ..research.opportunity import DETAIL, player_games
from .backtest import _final_rosters, final_weeks
from .optimize import BENCH, IR, SLOT
from .report import _con, _replay_frame, _roster_names, team_names

SKILL = ("QB", "RB", "WR", "TE")
def finished_weeks() -> list[int]:
    """Weeks that can be reviewed: finals archived, plus ESPN's projection history and nflverse usage for the week.

    Live weeks join once the Tuesday slot has saved the finals and pulled both downloads.
    """
    _refresh()
    hist = {int(p.stem[1:3]) for p in (DATA_DIR / "espn_history" / str(SEASON)).glob("w*-p0.json.gz")}
    # nflverse usage for every team that played (the files fill in game by game)
    teams = _games().group_by("week").agg(pl.col("team").n_unique().alias("n"))
    con = _con()
    played = dict(con.execute("SELECT week, 2 * COUNT(*) FROM pro_games WHERE season=? GROUP BY week", (SEASON,)).fetchall())
    used = {r["week"] for r in teams.iter_rows(named=True) if r["n"] >= played.get(r["week"], 99)}
    return sorted(final_weeks(con) & hist & used)


_version = None


def _refresh() -> None:
    """Drop cached frames when the downloads under them change (Tuesday's ESPN history and nflverse refresh)."""
    global _version
    files = [DATA_DIR / "espn_history" / "projections.parquet",
             *(NFLVERSE_DIR / d / f"{SEASON}.parquet" for d in ("ff_opportunity", "pbp", "snap_counts", "injuries"))]
    v = tuple(f.stat().st_mtime if f.exists() else None for f in files)
    if v != _version:
        for f in (_frame, _games, _snaps, role_reasons, _replay_frame):
            f.cache_clear()
        _version = v


@lru_cache(maxsize=1)
def _frame() -> pl.DataFrame:
    return _replay_frame()


@lru_cache(maxsize=1)
def _games() -> pl.DataFrame:
    return player_games([SEASON])


USAGE = ["targets", "team_targets", "carries", "team_carries", "i10", "team_i10", "air", "team_air", "pass_att"]
ESPN_LINE = ["p_rec", "p_rec_yds", "p_rec_td", "p_rush_yds", "p_rush_td", "p_pass_yds", "p_pass_td", "p_int"]


def _players(week: int) -> pl.DataFrame:
    f = _frame().filter(pl.col("week") == week).select(
        pl.col("espn_id").alias("player_id"), "name", "pos", "team", "opp", "proj", "gsis_id", *ESPN_LINE)
    g = _games()
    detail = [c for c in g.columns if c.endswith(("_xfp", "_fp")) or c.replace("_exp", "") in DETAIL]
    this = g.filter(pl.col("week") == week).select("gsis_id", "xfp", *USAGE, *detail)
    # his usual role: mean shares over his earlier games this season
    usual = (g.filter(pl.col("week") < week).group_by("gsis_id")
              .agg(pl.len().alias("usual_games"), *[pl.col(c).mean().alias(f"usual_{c}")
                                                    for c in ("tgt_share", "car_share", "i10_share", "air_share")]))
    # his team's usual volume (targets / carries per game) in earlier weeks
    team = (g.filter(pl.col("week") < week).unique(["team", "week"]).group_by("team")
             .agg(pl.col("team_targets").mean().alias("usual_team_targets"),
                  pl.col("team_carries").mean().alias("usual_team_carries")))
    return (f.join(this, on="gsis_id", how="left").join(usual, on="gsis_id", how="left")
             .join(team, on="team", how="left"))


@lru_cache(maxsize=1)
def _snaps() -> pl.DataFrame:
    """Offense snap share per player-week (snap counts are keyed by PFR id; map to gsis)."""
    ids = load("ff_playerids").select("pfr_id", "gsis_id").drop_nulls().unique("pfr_id")
    return (load("snap_counts", [SEASON]).filter(pl.col("game_type") == "REG")
            .join(ids, left_on="pfr_player_id", right_on="pfr_id")
            .select(pl.col("week").cast(pl.Int64), "gsis_id", "team", "position", pl.col("player").alias("name"),
                    pl.col("offense_pct").alias("snap")))


@lru_cache(maxsize=16)
def role_reasons(week: int) -> dict[str, list[dict]]:
    """Likely reasons a player's usage differed from ESPN's projection, from what happened in the game.

    Post-game guesses, not causes: snap share vs his usual, teammates out before or during the game,
    game script (final margin vs. the betting line), team play volume, and the next week's injury report.
    Each reason carries the direction it pushes usage (+1 more, -1 less), so the page shows only the ones
    that match the miss. -> gsis_id -> [{dir, text}]
    """
    sn = _snaps()
    this = sn.filter(pl.col("week") == week)
    usual = (sn.filter((pl.col("week") < week) & (pl.col("snap") > 0)).group_by("gsis_id")
               .agg(pl.col("snap").mean().alias("usual_snap"), pl.len().alias("n")))
    snaps = this.join(usual, on="gsis_id", how="left")
    snap_of = {r["gsis_id"]: r for r in snaps.iter_rows(named=True)}

    # in-game departures: a regular (usual 40%+ snaps) who played far less than usual, by team and position group
    group = lambda pos: "RB" if pos in ("RB", "FB") else "REC" if pos in ("WR", "TE") else pos  # noqa: E731
    left = {}
    for r in snaps.iter_rows(named=True):
        if (r["usual_snap"] or 0) >= 0.4 and (r["n"] or 0) >= 2 and r["snap"] <= r["usual_snap"] - 0.3:
            left.setdefault((r["team"], group(r["position"])), []).append(
                f"{r['name']} ({r['snap']:.0%} of snaps, usual {r['usual_snap']:.0%})")

    # next week's injury report: hurt in this game?
    inj = load("injuries", [SEASON]).filter(pl.col("week") == week + 1)
    hurt = {r["gsis_id"]: (r["report_primary_injury"] or "injury", r["report_status"] or r["practice_status"])
            for r in inj.iter_rows(named=True)}
    # pre-game: key teammates out (from that week's injury report and IR)
    from ..research.vacated import vacated
    vac = {r["team"]: r for r in vacated([SEASON]).filter(pl.col("week") == week).iter_rows(named=True) if r["vacated_who"]}
    # game script: final margin vs the closing line
    sch = load("schedules", [SEASON]).filter((pl.col("week") == week) & pl.col("home_score").is_not_null())
    script, wind = {}, {}
    for r in sch.iter_rows(named=True):
        hm = r["home_score"] - r["away_score"]
        script[r["home_team"]] = (hm, r["spread_line"], r["away_team"])
        script[r["away_team"]] = (-hm, -r["spread_line"], r["home_team"])
        if r["roof"] in ("outdoors", "open") and (r["wind"] or 0) >= 15:
            wind[r["home_team"]] = wind[r["away_team"]] = r["wind"]
    # team volume vs its earlier weeks: carries for RBs, targets for everyone else
    g = _games().unique(["team", "week"])
    vol = {}
    for col in ("team_targets", "team_carries"):
        now = {r["team"]: r[col] for r in g.filter(pl.col("week") == week).iter_rows(named=True)}
        us = {r["team"]: r[col] for r in g.filter(pl.col("week") < week).group_by("team").agg(pl.col(col).mean()).iter_rows(named=True)}
        vol[col] = {t: (now[t], us[t]) for t in now if us.get(t)}

    out: dict[str, list[dict]] = {}
    players = _frame().filter((pl.col("week") == week) & pl.col("pos").is_in(SKILL)).select("gsis_id", "name", "pos", "team")
    for p in players.iter_rows(named=True):
        gid, pos, team, rs = p["gsis_id"], p["pos"], p["team"], []
        sp = snap_of.get(gid)
        if sp and sp["usual_snap"] is not None and (sp["n"] or 0) >= 2:
            d = sp["snap"] - sp["usual_snap"]
            if d <= -0.2:
                h = hurt.get(gid)
                tail = f"; on next week's injury report ({h[0]}), so likely hurt in the game" if h else "; not on next week's injury report"
                rs.append({"dir": -1, "text": f"Played {sp['snap']:.0%} of snaps (usual {sp['usual_snap']:.0%}){tail}."})
            elif d >= 0.15:
                rs.append({"dir": 1, "text": f"Played {sp['snap']:.0%} of snaps (usual {sp['usual_snap']:.0%})."})
        elif sp is None and gid in hurt:
            rs.append({"dir": -1, "text": f"No snaps; on next week's injury report ({hurt[gid][0]})."})
        v = vac.get(team)
        if v:
            mates = [w for w, i in zip(v["vacated_who"].split("; "), v["vacated_ids"].split(",")) if i != gid]
            relevant = [m for m in mates if ("RB," in m) == (pos == "RB") or pos == "QB" or ("% of targets" in m and pos in ("WR", "TE", "RB"))]
            if relevant:
                rs.append({"dir": 1, "text": f"Out: {'; '.join(relevant)}."})
        mates_left = [m for m in left.get((team, group(pos)), []) if not m.startswith(p["name"] + " ")]
        if mates_left:
            rs.append({"dir": 1, "text": f"Left early: {'; '.join(mates_left)}."})
        if team in script and script[team][1] is not None:
            m, exp, opp = script[team]
            gap = m - exp
            res = f"beat {opp} by {m}" if m > 0 else f"lost to {opp} by {-m}" if m < 0 else f"tied {opp}"
            line = f"{exp:.1f}-pt favourites" if exp > 0 else f"{-exp:.1f}-pt underdogs" if exp < 0 else "a pick'em"
            if gap <= -10:  # trailed more than expected: more passing, fewer runs
                rs.append({"dir": -1 if pos == "RB" else 1, "text": f"{team} {res} as {line}; trailing teams throw more."})
            elif gap >= 10:
                rs.append({"dir": 1 if pos == "RB" else -1, "text": f"{team} {res} as {line}; leading teams run more."})
        col, what = ("team_carries", "ran") if pos == "RB" else ("team_targets", "threw")
        if team in vol[col]:
            now, us = vol[col][team]
            if now <= 0.8 * us or now >= 1.2 * us:
                rs.append({"dir": 1 if now > us else -1, "text": f"{team} {what} {now:.0f} times (usual {us:.0f})."})
        if team in wind:
            rs.append({"dir": 1 if pos == "RB" else -1, "text": f"{wind[team]:.0f} mph wind; wind cuts passing."})
        if rs:
            out[gid] = rs
    return out


def _why(gid: str | None, role: float | None, week: int) -> list[str]:
    if gid is None or role is None or abs(role) < 2:
        return []
    want = 1 if role > 0 else -1
    rs = [r["text"] for r in role_reasons(week).get(gid, []) if r["dir"] == want]
    return rs or ["No clear cause in snaps, injuries, game script or volume."]


def _parts(r: dict) -> list[dict]:
    """Expected vs actual points by part of the game, with the stats behind each, and ESPN's stat line."""
    g = lambda k: r.get(k) or 0.0  # noqa: E731
    espn = {"rec": g("p_rec") + 0.1 * g("p_rec_yds") + 6 * g("p_rec_td"),
            "rush": 0.1 * g("p_rush_yds") + 6 * g("p_rush_td"),
            "pass": 0.04 * g("p_pass_yds") + 4 * g("p_pass_td") - 2 * g("p_int")}
    stats = {"rec": [("rec", "receptions", "p_rec"), ("yds", "rec_yards_gained", "p_rec_yds"), ("TD", "rec_touchdown", "p_rec_td")],
             "rush": [("yds", "rush_yards_gained", "p_rush_yds"), ("TD", "rush_touchdown", "p_rush_td")],
             "pass": [("yds", "pass_yards_gained", "p_pass_yds"), ("TD", "pass_touchdown", "p_pass_td"),
                      ("INT", "pass_interception", "p_int")]}
    out = []
    for part, label in (("pass", "Passing"), ("rush", "Rushing"), ("rec", "Receiving")):
        if not (g(f"{part}_xfp") or g(f"{part}_fp") or espn[part] >= 0.5):
            continue
        out.append({"part": part, "label": label, "espn": espn[part], "xfp": g(f"{part}_xfp"), "fp": g(f"{part}_fp"),
                    "stats": [{"stat": s, "espn": g(e), "exp": g(f"{c}_exp"), "actual": g(c)} for s, c, e in stats[part]]})
    return out


def _row(r: dict, names: dict) -> dict:
    pos = r.get("pos") or (names.get(r["player_id"]) or (None, None))[1]
    name = r.get("name") or (names.get(r["player_id"]) or ("?",))[0]
    proj, actual = r.get("proj") or 0.0, r["actual"] or 0.0
    xfp = r.get("xfp") if pos in SKILL else None
    if pos in SKILL and xfp is None and proj:  # played no snaps with a touch, or inactive
        xfp = 0.0
    out = {"id": r["player_id"], "name": name, "pos": pos, "team": r.get("team"), "opp": r.get("opp"),
           "slot": SLOT.get(r["slot"], "BE" if r["slot"] == BENCH else "IR"), "proj": proj, "actual": actual,
           "xfp": xfp, "role": None if xfp is None else xfp - proj, "luck": None if xfp is None else actual - xfp,
           "usage": None, "parts": []}
    if xfp is not None and r.get("team_targets"):
        out["usage"] = {k: r.get(k) for k in USAGE} | {k: r.get(k) for k in r if k.startswith("usual_")}
        out["parts"] = _parts(r)
    out["why"] = _why(r.get("gsis_id"), out["role"], r["week"]) if xfp is not None else []
    return out


def _side(rows: list[dict]) -> dict:
    st = [p for p in rows if p["slot"] not in ("BE", "IR")]
    sk = [p for p in st if p["xfp"] is not None]
    return {"proj": sum(p["proj"] for p in st), "actual": sum(p["actual"] for p in st),
            "role": sum(p["role"] for p in sk), "luck": sum(p["luck"] for p in sk),
            "kdst": sum(p["actual"] - p["proj"] for p in st if p["xfp"] is None)}


def _bench_calls(me: list[dict]) -> list[dict]:
    """Bench players who outscored a starter they could have replaced (same position, or FLEX-eligible)."""
    flex = ("RB", "WR", "TE")
    out = []
    starters = [p for p in me if p["slot"] not in ("BE", "IR")]
    for b in (p for p in me if p["slot"] == "BE" and p["proj"] > 0):
        cands = [s for s in starters if s["pos"] == b["pos"] or (s["slot"] == "FLEX" and b["pos"] in flex)]
        if not cands:
            continue
        s = min(cands, key=lambda s: s["actual"])
        if b["actual"] - s["actual"] < 3:
            continue
        both = b["xfp"] is not None and s["xfp"] is not None
        if not both:
            why = "No opportunity data for this position: pure variance."
        elif b["proj"] < s["proj"] and b["xfp"] <= s["xfp"]:
            why = "Right call on the information: the starter had the bigger projection and the bigger role; the bench player was more efficient."
        elif b["xfp"] > s["xfp"] and b["proj"] < s["proj"]:
            why = "ESPN missed the role: the bench player was used more than the starter, against the projection."
        else:
            why = "The bench player had the higher projection too."
        out.append({"bench": b, "starter": s, "gain": b["actual"] - s["actual"], "why": why})
    return sorted(out, key=lambda x: -x["gain"])


def week_review(week: int, team_id: int | None = None) -> dict:
    con = _con()
    ros = _final_rosters(con, week)
    me = team_id or MY_TEAM_ID
    opp = ros.filter(pl.col("team_id") == me)["opp_team_id"][0]
    names = _roster_names(con, "replay", None, week)
    pl_ = _players(week)
    df = ros.filter(pl.col("team_id").is_in([me, opp])).join(pl_, on="player_id", how="left")
    df = df.with_columns(pl.lit(week).alias("week"))
    rows = {t: [_row(r, names) for r in df.filter(pl.col("team_id") == t).iter_rows(named=True)] for t in (me, opp)}
    tn = team_names()
    a, b = _side(rows[me]), _side(rows[opp])

    # league-wide: who got the biggest usage, and who did the most/least with it (rostered or not)
    lw = (pl_.filter(pl.col("pos").is_in(SKILL) & pl.col("xfp").is_not_null())
              .join(_frame().filter(pl.col("week") == week).select(pl.col("espn_id").alias("player_id"), "actual"),
                    on="player_id")
              .with_columns((pl.col("xfp") - pl.col("proj")).alias("role"), (pl.col("actual") - pl.col("xfp")).alias("luck")))
    pick = lambda d: [{"name": r["name"], "pos": r["pos"], "team": r["team"], "proj": r["proj"], "xfp": r["xfp"],  # noqa: E731
                       "actual": r["actual"], "role": r["role"], "luck": r["luck"],
                       "why": _why(r["gsis_id"], r["role"], week)} for r in d.iter_rows(named=True)]
    big = lw.filter(pl.col("xfp") >= 10)
    margins = {"proj_margin": a["proj"] - b["proj"], "role": a["role"] - b["role"], "luck": a["luck"] - b["luck"],
               "kdst": a["kdst"] - b["kdst"], "margin": a["actual"] - b["actual"]}
    return {"verdict": _verdict(margins),
        "week": week, "me": {"id": me, "name": tn.get(me)}, "opp": {"id": opp, "name": tn.get(opp)},
        "players": {"me": rows[me], "opp": rows[opp]}, "totals": {"me": a, "opp": b},
        "bench_calls": _bench_calls(rows[me]),
        "league": {"unlucky": pick(big.sort("luck").head(8)), "lucky": pick(big.sort("luck", descending=True).head(8)),
                   "role_up": pick(lw.sort("role", descending=True).head(8)), "role_down": pick(lw.filter(pl.col("proj") >= 8).sort("role").head(8))},
    }


def _verdict(w: dict) -> str:
    """One sentence: the projected margin, the final one, and which parts (8+ pts) moved it."""
    pm, m = w["proj_margin"], w["margin"]
    head = ("Projected as a toss-up" if abs(pm) < 1 else f"Projected to {'win' if pm >= 0 else 'lose'} by {abs(pm):.0f}") + \
        f"; {'won' if m > 0 else 'lost'} by {abs(m):.0f}"
    words = {"role": ("your starters got more work than ESPN expected", "your starters got less work than ESPN expected"),
             "luck": ("they did more with their touches than usual", "they did less with their touches than usual"),
             "kdst": ("K and D/ST beat their projections", "K and D/ST fell short of their projections")}
    big = sorted((k for k in words if abs(w[k]) >= 8), key=lambda k: -abs(w[k]))
    if not big:
        return head + ", about as projected."
    bits = [f"{words[k][0 if w[k] > 0 else 1]} ({w[k]:+.0f})" for k in big]
    return head + ": " + (", ".join(bits[:-1]) + " and " if len(bits) > 1 else "") + bits[-1] + "."


def season_review(team_id: int | None = None, weeks=None) -> dict:
    """Per finished week: this team's margin split into projection, role, luck and K/DST, with a verdict."""
    out = []
    for w in finished_weeks() if weeks is None else weeks:
        r = week_review(w, team_id)
        a, b = r["totals"]["me"], r["totals"]["opp"]
        out.append({"week": w, "opponent": r["opp"]["name"], "proj_margin": a["proj"] - b["proj"],
                    "role": a["role"] - b["role"], "luck": a["luck"] - b["luck"], "kdst": a["kdst"] - b["kdst"],
                    "margin": a["actual"] - b["actual"], "verdict": r["verdict"]})
    tot = {k: sum(w[k] for w in out) for k in ("proj_margin", "role", "luck", "kdst", "margin")}
    wins = sum(w["margin"] > 0 for w in out)
    fav = [w for w in out if w["proj_margin"] > 0]
    summary = (f"Over {len(out)} weeks: {wins}–{len(out) - wins}, outscoring opponents by {tot['margin']:.0f}. "
               f"{tot['proj_margin']:.0f} of that was already in the projections (stronger lineups on paper); "
               f"projected favourite {len(fav)} times, won {sum(w['margin'] > 0 for w in fav)}. "
               f"Usage vs ESPN {tot['role']:+.0f}, luck {tot['luck']:+.0f}, K and D/ST {tot['kdst']:+.0f}.")
    return {"weeks": out, "totals": tot, "summary": summary}
