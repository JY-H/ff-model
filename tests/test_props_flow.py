import json
from datetime import datetime, timezone
from pathlib import Path

from ff_model import db
from ff_model.archive.parse import parse_snapshot
from ff_model.sources import odds

SAMPLE = json.loads((Path(__file__).parent / "fixtures/props_kc_atl.json").read_text())


def test_props_fetched_once_and_parsed(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(db, "DATA_DIR", tmp_path)
    monkeypatch.setattr(odds.config, "odds_api_key", lambda: "test")
    con = db.connect()
    kick = datetime.fromisoformat(SAMPLE["commence_time"].replace("Z", "+00:00"))
    monkeypatch.setattr(odds, "fetch_events", lambda: [
        {"id": SAMPLE["id"], "commence_time": SAMPLE["commence_time"]},
        {"id": "later", "commence_time": "2026-11-16T01:20:00Z"}])
    calls = []

    def fake_call(con, sid, kind, path, params):
        calls.append((path, params["markets"]))
        con.execute("INSERT INTO odds_ledger (fetched_at, credits_remaining, kind) VALUES (?,?,?)",
                    (db.iso(db.utcnow()), 480, kind))
        return "url", params, 200, SAMPLE

    monkeypatch.setattr(odds, "_call", fake_call)
    sid = con.execute("INSERT INTO snapshots (season, week, started_at, status) VALUES"
                      " (2026, 10, 'x', 'running')").lastrowid
    start = datetime(2026, 11, 15, 16, 45, tzinfo=timezone.utc)
    for res in odds.fetch_props(con, sid, start, kick, 0, 0):
        db.save_raw(con, sid, "odds", "props", *res)
    assert calls == [(f"/events/{SAMPLE['id']}/odds", ",".join(odds.config.PROP_MARKETS))]

    # second run over the same window must not pay again
    assert list(odds.fetch_props(con, sid, start, kick, 0, 0)) == []

    parse_snapshot(con, sid, 10)
    rows = con.execute("SELECT market, COUNT(*) n FROM prop_lines GROUP BY market").fetchall()
    assert {r["market"] for r in rows} == set(odds.config.PROP_MARKETS)
    td = con.execute("SELECT * FROM prop_lines WHERE market='player_anytime_td' AND"
                     " player_name='Bijan Robinson' AND bookmaker='draftkings'").fetchone()
    assert td["outcome"] == "Yes" and td["price"] == -220

    # a snapshot in the next week sees none of last week's props
    from ff_model.model.report import _props
    nxt = con.execute("INSERT INTO snapshots (season, week, started_at, status) VALUES"
                      " (2026, 11, 'y', 'ok')").lastrowid
    assert _props(con, sid, "Bijan Robinson") and _props(con, nxt, "Bijan Robinson") == []
