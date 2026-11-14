"""Local-only web server for the recommendation page (http://127.0.0.1:8765).

GET  /                 the page
GET  /api/report       recommendation report for ?snapshot=ID (default: latest)
GET  /api/snapshots    recent snapshots to choose from
POST /api/snapshot     take a fresh snapshot now (ESPN + weather + injuries; no odds credits)
"""
from __future__ import annotations

import json
import logging
import math
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..db import connect

log = logging.getLogger("ff_model")
STATIC = Path(__file__).parent / "static"
_cache: dict[int, dict] = {}
_lock = threading.Lock()


def _clean(o):
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # keep the terminal quiet; errors still logged
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(_clean(obj), default=str).encode(), "application/json")

    def do_GET(self):
        url = urlparse(self.path)
        try:
            if url.path in ("/", "/index.html"):
                return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            if url.path == "/api/snapshots":
                con = connect()
                rows = con.execute("SELECT id, slot_key, started_at, week, status FROM snapshots WHERE status IN ('ok','partial')"
                                   " AND (slot_key IS NULL OR slot_key NOT LIKE '%backfill') ORDER BY id DESC LIMIT 30").fetchall()
                return self._json([dict(r) for r in rows])
            if url.path == "/api/report":
                from ..model.report import build_report
                sid = parse_qs(url.query).get("snapshot", [None])[0]
                sid = int(sid) if sid else None
                with _lock:
                    key = sid or connect().execute(
                        "SELECT MAX(id) FROM snapshots WHERE status IN ('ok','partial') AND"
                        " (slot_key IS NULL OR slot_key NOT LIKE '%backfill')").fetchone()[0]
                    if key not in _cache:
                        _cache[key] = build_report(key)
                return self._json(_cache[key])
            self._send(404, b"not found", "text/plain")
        except Exception as e:
            log.error("web error: %s\n%s", e, traceback.format_exc())
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self):
        if urlparse(self.path).path != "/api/snapshot":
            return self._send(404, b"not found", "text/plain")
        try:
            from ..archive import runner
            con = connect()
            runner.refresh_pro_schedule(con)
            sid = runner.snapshot(con, runner.current_week(con), with_odds=False)
            self._json({"snapshot_id": sid})
        except Exception as e:
            log.error("snapshot error: %s\n%s", e, traceback.format_exc())
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)


def serve(port: int = 8765, open_browser: bool = True) -> None:
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"recommendation page: {url}  (Ctrl-C to stop)")
    if open_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
