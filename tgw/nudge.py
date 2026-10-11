"""Gemma, running locally through Ollama, turns the numbers into one short nudge. Nothing leaves the laptop.

All the reasoning (which window, which spot, what changes later in the week) happens in code first; the small model
only has to phrase a handful of facts. That keeps it honest and keeps it fast on a laptop.
"""
import json
from datetime import date, timedelta

import requests

from tgw import DATA

OLLAMA = "http://127.0.0.1:11434/api/chat"
MODEL = "gemma4:e4b"

DEFAULT_SPOTS = [
    {"name": "the Main Quad", "min_hours": 0, "note": "the lawn between classes"},
    {"name": "the Japan House garden and the Arboretum", "min_hours": 1.5, "note": "south campus, walk or bike"},
    {"name": "Crystal Lake Park in Urbana", "min_hours": 1.5, "note": "lake loop, bus or bike"},
    {"name": "Allerton Park in Monticello", "min_hours": 5, "weekend_only": True, "note": "needs a car or a friend with one"},
]

SYSTEM = """You write a short nudge that gets a student outside, from the facts the user gives you.
Write exactly 3 sentences, under 75 words, in second person. Warm and a bit dry. No exclamation marks, no markdown, at most one emoji.
1. When to go, how it will feel, and where (the spot).
2. Why this one counts: what happens later in the week (if given), and how many days like this are left before New Year
   by their own taste versus what a generic weather app would say.
3. "Shoes on at <time>." using the given time, or "Shoes on now." if the time is "now".
Use only the given facts. Do not add any other numbers, places or weather."""


def spots() -> list[dict]:
    f = DATA / "spots.json"
    return json.loads(f.read_text()) if f.exists() else DEFAULT_SPOTS


def pick_spot(window: dict) -> dict:
    weekend = date.fromisoformat(window["date"]).weekday() >= 5
    fits = [s for s in spots() if s["min_hours"] <= window["hours"] and (weekend or not s.get("weekend_only"))]
    biggest = max(s["min_hours"] for s in fits)
    options = [s for s in fits if s["min_hours"] == biggest]
    return options[date.fromisoformat(window["date"]).toordinal() % len(options)]  # vary between equal options


def coming(week_by_day: list[dict], after: str) -> str | None:
    """The first run of days after the window's day with no hours you'd go out in, and why."""
    reasons, started = [], False
    for d in week_by_day:
        if not started and d["date"] <= after:
            continue
        started = True
        if d["go_hours"] > 0:
            if reasons:
                break
            continue
        why = []
        if d["storm_hours"]:
            why.append("thunderstorms")
        if d["rain_mm"] >= 5:
            why.append(f"{d['rain_mm']:.0f} mm of rain")
        if d["feels_c_range"][1] < 10:
            why.append(f"a feels-like high of {d['feels_c_range'][1]}°C")
        reasons.append(f"{d['weekday']}: {', '.join(why) or 'nothing you would go out in'}")
    return "; ".join(reasons) or None


def brief(facts: dict) -> dict | None:
    w = facts.get("best_window")
    if not w:
        return None
    today = date.today()
    d = date.fromisoformat(w["date"])
    when = "today" if d == today else "tomorrow" if d == today + timedelta(days=1) else d.strftime("%A")
    spot = pick_spot(w)
    sky = "sunny" if w["cloud_pct"] < 25 else "partly cloudy" if w["cloud_pct"] < 70 else "overcast"
    wind = "calm" if w["wind_kmh"] < 10 else "breezy" if w["wind_kmh"] < 25 else "windy"
    wet = ", with some rain" if w.get("rain_mm", 0) >= 1 else ""
    return {
        "Go": f"{when} ({d.strftime('%A')}), {w['best_hour'] if w['hours'] > 3 else w['start']}–{w['end']}"
        + (f", sunset is at {facts['sunset_today']}" if when == "today" and facts.get("sunset_today") else ""),
        "Feels like": f"{w['feels_c']}°C, {sky}, {wind}{wet}",
        "Spot": f"{spot['name']} ({spot['note']})",
        "Later this week": coming(facts.get("week_by_day", []), w["date"]),
        "Days like this left before New Year in a typical year, by your own taste": f"about {round(facts['good_days_left']['average'])}",
        'What a generic "nice weather" rule would say': f"about {round(facts['good_days_left']['generic_average'])}",
        "Shoes on at": w["shoes_on"],
    }


def write(facts: dict, timeout: int = 180) -> tuple[str, str]:
    """Returns (message, writer). Writer is the model name, or 'template' if Ollama is not running."""
    b = brief(facts)
    if b is None:
        return "Nothing this week clears your bar. Good week for the problem set; I'll look again tomorrow.", "template"
    try:
        r = requests.post(
            OLLAMA,
            json={
                "model": MODEL,
                "stream": False,
                "think": True,  # a moment of thinking makes the small model follow the facts much more closely
                "options": {"temperature": 0.5},
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": "\n".join(f"- {k}: {v}" for k, v in b.items() if v)},
                ],
            },
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()["message"]["content"].strip(), MODEL
    except (requests.RequestException, KeyError):
        return _template(b), "template"


def _template(b: dict) -> str:
    msg = f"Go {b['Go']}, feels like {b['Feels like']}: {b['Spot'].split(' (')[0]}."
    if b.get("Later this week"):
        msg += f" Later this week: {b['Later this week']}."
    shoes = "now" if b["Shoes on at"] == "now" else f"at {b['Shoes on at']}"
    return msg + f" Shoes on {shoes}."
