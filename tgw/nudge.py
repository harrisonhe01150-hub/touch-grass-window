"""Gemma, running locally through Ollama, turns the numbers into one short nudge. Nothing leaves the laptop."""
import json
from pathlib import Path

import requests

DATA = Path(__file__).resolve().parent.parent / "data"
OLLAMA = "http://127.0.0.1:11434/api/chat"
MODEL = "gemma4:e4b"

DEFAULT_SPOTS = [
    {"name": "Main Quad", "fits": "any window, even 30 minutes", "note": "the lawn between classes"},
    {"name": "Japan House garden and the Arboretum", "fits": "1-2 hours", "note": "south campus, walk or bike"},
    {"name": "Crystal Lake Park, Urbana", "fits": "1-2 hours", "note": "lake loop, bus or bike"},
    {"name": "Allerton Park, Monticello", "fits": "half a day", "note": "needs a car or a friend with one"},
]

SYSTEM = """You write one short nudge that gets a student to go outside.
Rules:
- Use ONLY the facts in the JSON. Never invent numbers, places, or weather.
- Name the exact best window (day, start-end time). Pick ONE spot whose "fits" matches the window length:
  a window of 3+ hours deserves a bigger place than the Main Quad.
- If the best window is today, say so plainly, and mention sunset_today if it is the reason to hurry.
- Say why it matters with good_days_left, in plain words, e.g. "in a typical year you only get about N more days like this
  before the year ends". Round N to a whole number. Do not use the phrase "typical-year average".
- Under 80 words. Second person. Warm, a bit dry, no lecture.
- End with one tiny action with a concrete clock time: shoes on 10 minutes before the window starts, or "now" if it already has.
- Plain text, no markdown, at most one emoji."""


def spots() -> list[dict]:
    f = DATA / "spots.json"
    return json.loads(f.read_text()) if f.exists() else DEFAULT_SPOTS


def write(facts: dict, timeout: int = 120) -> tuple[str, str]:
    """Returns (message, source). Source is the model name, or 'template' if Ollama is not running."""
    facts = {**facts, "spots": spots()}
    try:
        r = requests.post(
            OLLAMA,
            json={
                "model": MODEL,
                "stream": False,
                "think": False,
                "options": {"temperature": 0.6},
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": json.dumps(facts, ensure_ascii=False)},
                ],
            },
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()["message"]["content"].strip(), MODEL
    except (requests.RequestException, KeyError):
        return _template(facts), "template"


def _template(facts: dict) -> str:
    w = facts.get("best_window")
    if not w:
        return "No window this week clears your bar. Good week to finish the problem set. I'll check again tomorrow."
    return f"Best window: {w['day']} {w['start']}-{w['end']}, feels like {w['feels_c']}°C. Go to {spots()[0]['name']}."
