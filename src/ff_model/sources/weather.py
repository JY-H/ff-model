"""Kickoff-window weather forecasts from Open-Meteo (free, no key).

One request per snapshot covers every upcoming game: Open-Meteo accepts a list
of coordinates. In the US its default model blend uses NOAA's 3 km HRRR for the
short range. Forecasts can't be fetched after the fact, so every response is
archived.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import requests

log = logging.getLogger(__name__)

URL = "https://api.open-meteo.com/v1/forecast"
HOURLY = ["temperature_2m", "wind_speed_10m", "wind_gusts_10m", "precipitation", "precipitation_probability"]
GAME_HOURS = 3  # average over kickoff .. kickoff + 3h


def fetch(games: list[dict]):
    """games: [{game_id, lat, lon, kickoff: datetime}]. Returns (url, params, status, body) or None."""
    if not games:
        return None
    start = min(g["kickoff"] for g in games).astimezone(timezone.utc).date()
    end = (max(g["kickoff"] for g in games) + timedelta(hours=GAME_HOURS + 1)).astimezone(timezone.utc).date()
    params = {"latitude": ",".join(f"{g['lat']:.4f}" for g in games),
              "longitude": ",".join(f"{g['lon']:.4f}" for g in games),
              "hourly": ",".join(HOURLY), "wind_speed_unit": "mph", "temperature_unit": "fahrenheit",
              "precipitation_unit": "inch", "timezone": "GMT",
              "start_date": start.isoformat(), "end_date": end.isoformat()}
    try:
        r = requests.get(URL, params=params, timeout=30,
                         headers={"User-Agent": "ff-model-archiver (personal use)"})
    except requests.RequestException as e:
        log.warning("weather request failed: %s", type(e).__name__)
        return None
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else None
    log.info("weather HTTP %d for %d games", r.status_code, len(games))
    return URL, {**params, "game_ids": [g["game_id"] for g in games]}, r.status_code, body


def summarize(body, games: list[dict]) -> list[dict]:
    """Per game: mean temp/wind over the game window, max gust, total precip, max precip chance."""
    locs = body if isinstance(body, list) else [body]
    out = []
    for g, loc in zip(games, locs):
        h = loc.get("hourly") or {}
        times = [datetime.fromisoformat(t).replace(tzinfo=timezone.utc) for t in h.get("time", [])]
        k0 = g["kickoff"].astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
        idx = [i for i, t in enumerate(times) if k0 <= t <= k0 + timedelta(hours=GAME_HOURS)]
        if not idx:
            continue

        def vals(key):
            return [h[key][i] for i in idx if h.get(key) and h[key][i] is not None]

        def mean(key):
            v = vals(key)
            return sum(v) / len(v) if v else None

        out.append({"game_id": g["game_id"], "temp_f": mean("temperature_2m"),
                    "wind_mph": mean("wind_speed_10m"),
                    "gust_mph": max(vals("wind_gusts_10m"), default=None),
                    "precip_in": sum(vals("precipitation")) if vals("precipitation") else None,
                    "precip_prob": max(vals("precipitation_probability"), default=None)})
    return out
