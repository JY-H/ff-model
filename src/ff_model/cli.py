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
    args = ap.parse_args(argv)
    _setup_logging(args.verbose or args.cmd != "tick")

    if args.cmd == "tick":
        runner.tick()
        return
    if args.cmd == "install-launchd":
        install_launchd()
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
    elif args.cmd == "reparse":
        ids = args.snapshot_ids or [r["id"] for r in con.execute("SELECT id FROM snapshots")]
        for sid in ids:
            week = con.execute("SELECT week FROM snapshots WHERE id=?", (sid,)).fetchone()["week"]
            parse_snapshot(con, sid, week)
        print(f"reparsed {len(ids)} snapshots")


if __name__ == "__main__":
    main(sys.argv[1:])
