"""Local-only web server for the recommendation page (http://127.0.0.1:8765).

GET  /                 the page
GET  /api/weeks        weeks (live snapshots or replay) and teams
GET  /api/report       ?week=&team=&snapshot=  recommendation report
GET  /api/context      ?week=&snapshot=        raw context for the whole week
GET  /api/tracking     ?team=                  season tracking
POST /api/whatif       same query + JSON {force_in, force_out, proj_delta}
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
_cache: dict[tuple, dict] = {}
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

    def _q(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        return u.path, {k: int(v) for k, v in q.items() if v not in ("", "null")}

    def do_GET(self):
        path, q = self._q()
        try:
            if path in ("/", "/index.html"):
                return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            from ..model import report as R
            if path == "/api/weeks":
                return self._json({"weeks": R.weeks_available(),
                                   "teams": [{"id": k, "name": v} for k, v in sorted(R.team_names().items(), key=lambda kv: kv[1].lower())],
                                   "me": R.MY_TEAM_ID})
            if path == "/api/report":
                key = (q.get("week"), q.get("team"), q.get("snapshot"))
                with _lock:
                    if key not in _cache:
                        _cache[key] = R.build_report(week=key[0], team_id=key[1], snapshot_id=key[2])
                return self._json(_cache[key])
            if path == "/api/context":
                return self._json(R.week_context(q.get("week"), q.get("snapshot")))
            if path == "/api/tracking":
                return self._json(R.season_tracking(q.get("team")))
            self._send(404, b"not found", "text/plain")
        except Exception as e:
            log.error("web error: %s\n%s", e, traceback.format_exc())
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self):
        path, q = self._q()
        try:
            if path == "/api/snapshot":
                from ..archive import runner
                con = connect()
                runner.refresh_pro_schedule(con)
                sid = runner.snapshot(con, runner.current_week(con), with_odds=False)
                _cache.clear()
                return self._json({"snapshot_id": sid})
            if path == "/api/whatif":  # recompute with overrides; never cached
                from ..model.report import build_report
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                return self._json(build_report(week=q.get("week"), team_id=q.get("team"), snapshot_id=q.get("snapshot"),
                                               overrides=body))
            self._send(404, b"not found", "text/plain")
        except Exception as e:
            log.error("web error: %s\n%s", e, traceback.format_exc())
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
