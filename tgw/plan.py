"""Turn hourly go-outside probabilities into concrete windows, and say how rare a good day is."""
import json
from datetime import date

import numpy as np
import pandas as pd

from tgw import DATA

GO = 0.5  # p(go) threshold. Rain is left to your taste; only thunderstorms veto an hour (lightning).


def free_hours() -> dict:
    """Optional data/free_hours.json: {"Mon": [[12, 14], [16, 18]], ...}. Missing day = free all day."""
    f = DATA / "free_hours.json"
    return json.loads(f.read_text()) if f.exists() else {}


def _is_free(t: pd.Timestamp, free: dict) -> bool:
    slots = free.get(t.strftime("%a"))
    return slots is None or any(a <= t.hour < b for a, b in slots)


def windows(fc: pd.DataFrame, p, sunsets: dict | None = None, now: pd.Timestamp | None = None) -> list[dict]:
    """Runs of consecutive daylight, free, storm-free hours the person would say yes to. Ends are capped at sunset."""
    now = now or pd.Timestamp.now()
    sunsets = sunsets or {}
    free = free_hours()
    fc = fc.assign(p=p)
    ok = (fc["p"] >= GO) & (fc["is_day"] == 1) & ~fc["thunder"] & (fc["time"] >= now.floor("h"))
    ok &= fc["time"].map(lambda t: _is_free(t, free))
    out, run = [], []
    for i, good in enumerate(ok.tolist() + [False]):
        if good and (not run or fc["time"].iloc[i] - fc["time"].iloc[run[-1]] == pd.Timedelta("1h")):
            run.append(i)
            continue
        if run:
            w = fc.iloc[run]
            start, end = w["time"].iloc[0], w["time"].iloc[-1] + pd.Timedelta("1h")
            day = start.date().isoformat()
            if day in sunsets:
                end = min(end, pd.Timestamp(sunsets[day]))
            start_shown = max(start, now)
            # In a long window, aim for a near-best hour, closest to mid-afternoon (add data/free_hours.json to do better).
            near_best = w[w["p"].round(1) == w["p"].round(1).max()]
            peak = near_best.loc[(near_best["hour"] - 15).abs().idxmin(), "time"] if len(w) > 3 else start
            go_at = max(peak, start_shown)
            if end - start_shown >= pd.Timedelta("20min"):
                out.append(
                    {
                        "day": start.strftime("%a %b %d"),
                        "date": day,
                        "start": start.strftime("%H:%M"),
                        "end": end.strftime("%H:%M"),
                        "hours": round((end - start).total_seconds() / 3600, 1),
                        "already_started": bool(start <= now),
                        "best_hour": go_at.strftime("%H:%M"),
                        "shoes_on": "now" if go_at <= now else (go_at - pd.Timedelta("10min")).strftime("%H:%M"),
                        "p": round(float(w["p"].mean()), 2),
                        "feels_c": round(float(w["feels_c"].mean())),
                        "wind_kmh": round(float(w["wind_kmh"].mean())),
                        "cloud_pct": round(float(w["cloud_pct"].mean())),
                        "rain_mm": round(float(w["rain_mm"].sum()), 1),
                    }
                )
        run = [i] if good else []
    return out


def week(fc: pd.DataFrame, p) -> list[dict]:
    """One line per day: how many daylight hours clear the bar, and what the weather does."""
    fc = fc.assign(p=p)
    day = fc[fc["is_day"] == 1]
    out = []
    for d, g in day.groupby(day["time"].dt.date):
        out.append(
            {
                "day": pd.Timestamp(d).strftime("%a %b %d"),
                "date": d.isoformat(),
                "weekday": pd.Timestamp(d).strftime("%A"),
                "go_hours": int(((g["p"] >= GO) & ~g["thunder"]).sum()),
                "storm_hours": int(g["thunder"].sum()),
                "feels_c_range": [round(float(g["feels_c"].min())), round(float(g["feels_c"].max()))],
                "rain_mm": round(float(g["rain_mm"].sum()), 1),
                "max_rain_prob": int(g["rain_prob"].max()),
            }
        )
    return out


def good_days_left(history: pd.DataFrame, p_go, today: date | None = None) -> dict:
    """In a typical past year, how many days from tomorrow to Dec 31 had 2+ daytime hours that clear the bar?

    p_go: function from a weather frame to probabilities (the person's taste, or a generic rule).
    """
    today = today or date.today()
    h = history[(history["hour"].between(9, 18)) & (history["is_day"] == 1)].copy()
    md = h["time"].dt.month * 100 + h["time"].dt.day
    h = h[md > today.month * 100 + today.day]
    h["go"] = np.asarray(p_go(h)) >= GO
    per_day = h.groupby(h["time"].dt.date)["go"].sum()
    good = (per_day >= 2).groupby(lambda d: d.year).sum()
    return {
        "per_year": {int(y): int(n) for y, n in good.items()},
        "average": round(float(good.mean()), 1),
        "min": int(good.min()),
        "max": int(good.max()),
        "from": today.isoformat(),
    }
