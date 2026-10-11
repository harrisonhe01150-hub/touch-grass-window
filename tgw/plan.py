"""Turn hourly go-outside probabilities into concrete windows, and say how rare a good day is."""
import json
from datetime import date

import pandas as pd

from tgw import DATA

GO, RAIN_RISK = 0.5, 60  # p(go) threshold; precipitation probability (%) that vetoes an hour


def free_hours() -> dict:
    """Optional data/free_hours.json: {"Mon": [[12, 14], [16, 18]], ...}. Missing day = free all day."""
    f = DATA / "free_hours.json"
    return json.loads(f.read_text()) if f.exists() else {}


def _is_free(t: pd.Timestamp, free: dict) -> bool:
    slots = free.get(t.strftime("%a"))
    return slots is None or any(a <= t.hour < b for a, b in slots)


def windows(fc: pd.DataFrame, p, now: pd.Timestamp | None = None) -> list[dict]:
    """Runs of consecutive daylight, free, low-rain-risk hours the person would say yes to."""
    now = now or pd.Timestamp.now().floor("h")
    free = free_hours()
    fc = fc.assign(p=p)
    ok = (fc["p"] >= GO) & (fc["is_day"] == 1) & (fc["rain_prob"] < RAIN_RISK) & (fc["time"] >= now)
    ok &= fc["time"].map(lambda t: _is_free(t, free))
    out, run = [], []
    for i, good in enumerate(ok.tolist() + [False]):
        if good and (not run or fc["time"].iloc[i] - fc["time"].iloc[run[-1]] == pd.Timedelta("1h")):
            run.append(i)
            continue
        if run:
            w = fc.iloc[run]
            out.append(
                {
                    "day": w["time"].iloc[0].strftime("%a %b %d"),
                    "date": w["time"].iloc[0].date().isoformat(),
                    "start": w["time"].iloc[0].strftime("%H:%M"),
                    "end": (w["time"].iloc[-1] + pd.Timedelta("1h")).strftime("%H:%M"),
                    "hours": len(w),
                    "p": round(float(w["p"].mean()), 2),
                    "feels_c": round(float(w["feels_c"].mean())),
                    "wind_kmh": round(float(w["wind_kmh"].mean())),
                    "cloud_pct": round(float(w["cloud_pct"].mean())),
                }
            )
        run = [i] if good else []
    return out


def good_days_left(history: pd.DataFrame, taste, today: date | None = None) -> dict:
    """In a typical past year, how many days from tomorrow to Dec 31 had 2+ hours this person would go out in?"""
    today = today or date.today()
    h = history[(history["hour"].between(9, 18)) & (history["is_day"] == 1)].copy()
    md = h["time"].dt.month * 100 + h["time"].dt.day
    h = h[md > today.month * 100 + today.day]
    h["go"] = taste.p_go(h) >= GO
    per_day = h.groupby(h["time"].dt.date)["go"].sum()
    good = (per_day >= 2).groupby(lambda d: d.year).sum()
    return {
        "per_year": {int(y): int(n) for y, n in good.items()},
        "average": round(float(good.mean()), 1),
        "min": int(good.min()),
        "max": int(good.max()),
        "from": today.isoformat(),
    }
