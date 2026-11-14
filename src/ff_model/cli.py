"""ff-archive: command-line entry point."""
from __future__ import annotations

import argparse
import logging
import sys
from logging.handlers import RotatingFileHandler

from .archive import runner
from .archive.launchd import install_launchd
from .archive.parse import parse_snapshot
from .config import LOG_DIR
from .db import connect


def _setup_logging(verbose: bool) -> None:
    LOG_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh = RotatingFileHandler(LOG_DIR / "archiver.log", maxBytes=2_000_000, backupCount=5)
    fh.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(fh)
    if verbose:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)
    # urllib3 debug logs would include request URLs (the odds key is a query param).
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="ff-archive")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tick", help="run any due snapshot slot (called by launchd every 5 min)")
    sp = sub.add_parser("snapshot", help="take a manual snapshot now")
    sp.add_argument("--week", type=int)
    sp.add_argument("--odds", action="store_true", help="also fetch odds (costs 2 credits)")
    sp.add_argument("--prev-week", action="store_true", help="also archive last week's finals")
    sub.add_parser("status", help="show upcoming/missed slots and recent snapshots")
    sub.add_parser("install-launchd", help="write the LaunchAgent plist")
    rp = sub.add_parser("reparse", help="rebuild derived tables from raw responses")
    rp.add_argument("snapshot_ids", type=int, nargs="*")
    bf = sub.add_parser("backfill-finals", help="archive final lineups/points for past weeks")
    bf.add_argument("weeks", type=int, nargs="+")
    eh = sub.add_parser("espn-history", help="backfill ESPN weekly projections 2019-2025")
    eh.add_argument("--seasons", type=int, nargs="*")
    eh.add_argument("--weeks", type=int, nargs="*", help="only these weeks (e.g. completed weeks this season)")
    sub.add_parser("fit-marginals", help="fit player distributions on 2019-2025 and save")
    sub.add_parser("fit-correlation", help="estimate role-pair correlations on 2019-2025 and save")
    sub.add_parser("screen-factors", help="regenerate the factor evidence table used by the web page")
    bs = sub.add_parser("backtest-synth", help="walk-forward synthetic-league backtest, 2022-2025 (long)")
    bs.add_argument("--leagues", type=int, default=10)
    bs.add_argument("--draws", type=int, default=4000)
    bs.add_argument("--variant", default="model", choices=["model", "espn_mean", "espn_mean_ctx", "generic", "shrunk"])
    bs.add_argument("--seasons", type=int, nargs="*", default=[2022, 2023, 2024, 2025])
    wb = sub.add_parser("web", help="open the recommendation page (local only)")
    wb.add_argument("--port", type=int, default=8765)
    wb.add_argument("--no-browser", action="store_true")
    rc = sub.add_parser("recommend", help="best lineup by P(win) for my matchup")
    rc.add_argument("--snapshot", type=int, help="default: latest snapshot")
    rc.add_argument("--draws", type=int, default=20000)
    rk = sub.add_parser("rankings", help="write a shareable all-player rankings page (no league details)")
    rk.add_argument("--snapshot", type=int, help="default: latest snapshot")
    np_ = sub.add_parser("nflverse", help="download nflverse history (current season refreshed)")
    np_.add_argument("--seasons", type=int, nargs="*", help="default: 2012 through this season")
    np_.add_argument("--force", action="store_true", help="re-download completed seasons too")
    args = ap.parse_args(argv)
    _setup_logging(args.verbose or args.cmd != "tick")

    if args.cmd == "tick":
        runner.tick()
        return
    if args.cmd == "install-launchd":
        install_launchd()
        return
    if args.cmd == "espn-history":
        import polars as pl

        from .research import espn_history
        espn_history.download(args.seasons, args.weeks)
        df = espn_history.parse()
        print(df.group_by("season").agg(weeks=pl.col("week").n_unique(), rows=pl.len()).sort("season"))
        return
    if args.cmd == "fit-marginals":
        from .model.marginals import MODEL_PATH, fit_live
        fit_live()
        print(f"saved {MODEL_PATH}")
        return
    if args.cmd == "fit-correlation":
        from .model.correlation import CORR_PATH, fit_live
        fit_live()
        print(f"saved {CORR_PATH}")
        return
    if args.cmd == "screen-factors":
        from .research.factor_tests import run_screen
        print(f"{run_screen().height} rows")
        return
    if args.cmd == "backtest-synth":
        from .model.backtest_synth import run as run_synth
        run_synth(seasons=args.seasons, leagues=args.leagues, n=args.draws, variant=args.variant)
        return
    if args.cmd == "web":
        from .web.server import serve
        serve(args.port, not args.no_browser)
        return
    if args.cmd == "recommend":
        from .model.optimize import format_text, recommend
        print(format_text(recommend(args.snapshot, n=args.draws)))
        return
    if args.cmd == "rankings":
        from .model.rankings import write
        print(write(args.snapshot))
        return
    if args.cmd == "nflverse":
        import polars as pl

        from .research import nflverse
        nflverse.pull(args.seasons, force=args.force)
        print(nflverse.inventory().group_by("dataset").agg(
            min_season=pl.col("season").min(), max_season=pl.col("season").max(),
            rows=pl.col("rows").sum()).sort("dataset"))
        return
    con = connect()
    runner.refresh_pro_schedule(con)
    if args.cmd == "snapshot":
        week = args.week or runner.current_week(con)
        sid = runner.snapshot(con, week, with_odds=args.odds, prev_week=args.prev_week)
        r = con.execute("SELECT status, error FROM snapshots WHERE id=?", (sid,)).fetchone()
        print(f"snapshot {sid}: {r['status']}" + (f" ({r['error']})" if r["error"] else ""))
    elif args.cmd == "status":
        print(runner.status_report(con))
    elif args.cmd == "backfill-finals":
        for w in args.weeks:
            sid = runner.backfill_finals(con, w)
            print(f"week {w}: " + (f"snapshot {sid}" if sid else "already archived"))
    elif args.cmd == "reparse":
        ids = args.snapshot_ids or [r["id"] for r in con.execute("SELECT id FROM snapshots")]
        for sid in ids:
            week = con.execute("SELECT week FROM snapshots WHERE id=?", (sid,)).fetchone()["week"]
            parse_snapshot(con, sid, week)
        print(f"reparsed {len(ids)} snapshots")


if __name__ == "__main__":
    main(sys.argv[1:])
