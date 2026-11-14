"""Write the LaunchAgent that runs `ff-archive tick` every 5 minutes."""
from __future__ import annotations

import plistlib
from pathlib import Path

from ..config import LOG_DIR, ROOT

LABEL = "local.ff-model.archiver"
PLIST = Path.home() / "Library/LaunchAgents" / f"{LABEL}.plist"


def install_launchd() -> None:
    exe = ROOT / ".venv/bin/ff-archive"
    plist = {
        "Label": LABEL,
        # caffeinate keeps the Mac awake until the snapshot finishes
        "ProgramArguments": ["/usr/bin/caffeinate", "-i", str(exe), "tick"],
        "WorkingDirectory": str(ROOT),
        # Calendar intervals (unlike StartInterval) fire once on wake if missed.
        "StartCalendarInterval": [{"Minute": m} for m in range(0, 60, 5)],
        "StandardOutPath": str(LOG_DIR / "launchd.out.log"),
        "StandardErrorPath": str(LOG_DIR / "launchd.err.log"),
    }
    PLIST.parent.mkdir(parents=True, exist_ok=True)
    with open(PLIST, "wb") as f:
        plistlib.dump(plist, f)
    print(f"wrote {PLIST}\nload with:  launchctl bootstrap gui/$(id -u) {PLIST}")
