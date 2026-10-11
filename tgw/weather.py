"""Open weather data from Open-Meteo (CC BY 4.0). Cached on disk so the app works offline."""
import json
from datetime import date

import pandas as pd
import requests

from tgw import DATA

LAT, LON, TZ = 40.1106, -88.2272, "America/Chicago"  # Champaign-Urbana, IL

HOURLY = [
    "temperature_2m",
    "apparent_temperature",
    "precipitation",
    "wind_speed_10m",
    "cloud_cover",
    "relative_humidity_2m",
    "is_day",
]
RENAME = {
    "temperature_2m": "temp_c",
    "apparent_temperature": "feels_c",
    "precipitation": "rain_mm",
    "wind_speed_10m": "wind_kmh",
    "cloud_cover": "cloud_pct",
    "relative_humidity_2m": "humidity_pct",
}


def _frame(hourly: dict) -> pd.DataFrame:
    df = pd.DataFrame(hourly).rename(columns=RENAME)
    df["time"] = pd.to_datetime(df["time"])
    df["hour"] = df["time"].dt.hour
    return df.dropna(subset=list(RENAME.values()))


def history(start: str = "2016-01-01", end: str = "2025-12-31") -> pd.DataFrame:
    """Ten years of hourly Champaign weather. Downloaded once, then read from data/history.csv."""
    cache = DATA / "history.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["time"])
    r = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": LAT,
            "longitude": LON,
            "start_date": start,
            "end_date": end,
            "hourly": ",".join(HOURLY),
            "timezone": TZ,
        },
        timeout=120,
    )
    r.raise_for_status()
    df = _frame(r.json()["hourly"])
    DATA.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def forecast(offline: bool = False) -> tuple[pd.DataFrame, dict]:
    """Next 7 days, hourly, plus sunrise/sunset. Falls back to the last cached copy when offline."""
    cache = DATA / "forecast.json"
    payload = None
    if not offline:
        try:
            r = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": LAT,
                    "longitude": LON,
                    "hourly": ",".join(HOURLY + ["precipitation_probability", "weather_code"]),
                    "daily": "sunrise,sunset",
                    "timezone": TZ,
                    "forecast_days": 7,
                },
                timeout=30,
            )
            r.raise_for_status()
            payload = r.json()
            payload["fetched"] = date.today().isoformat()
            DATA.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(payload))
        except requests.RequestException:
            payload = None
    if payload is None:
        if not cache.exists():
            raise RuntimeError("No internet and no cached forecast yet. Run once online first.")
        payload = json.loads(cache.read_text())
    hourly = dict(payload["hourly"])
    hourly.setdefault("weather_code", [0] * len(hourly["time"]))  # older caches
    df = _frame(hourly).rename(columns={"precipitation_probability": "rain_prob"})
    df["thunder"] = df["weather_code"].between(95, 99)  # WMO codes 95-99: thunderstorm
    daily = {d: (sr, ss) for d, sr, ss in zip(payload["daily"]["time"], payload["daily"]["sunrise"], payload["daily"]["sunset"])}
    return df, {"daily": daily, "fetched": payload.get("fetched", "?")}
