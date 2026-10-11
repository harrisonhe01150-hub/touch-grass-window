"""Behaviour tests for the guards added after real bugs on day one. Fast: no TabPFN, no Ollama.

Run: python -m pytest -q
"""
import os
import tempfile

os.environ["TGW_DATA"] = tempfile.mkdtemp()  # never touch real labels

import pandas as pd  # noqa: E402

from tgw import nudge, plan, taste  # noqa: E402


def test_double_submitted_card_counts_once_latest_wins():
    rows = [{"time": "2016-09-16 12:00:00", **{f: 1.0 for f in taste.FEATURES}, "go": g} for g in (1, 1, 0)]
    pd.DataFrame(rows).to_csv(taste.LABELS, index=False)
    labels = taste.load_labels()
    assert len(labels) == 1 and labels["go"].iloc[0] == 0


def _forecast(day="2026-10-12"):
    t = pd.date_range(f"{day} 07:00", periods=13, freq="h")
    return pd.DataFrame({"time": t, "hour": t.hour, "is_day": [0] + [1] * 11 + [0], "thunder": False,
                         "feels_c": 20.0, "wind_kmh": 10.0, "cloud_pct": 10.0, "rain_mm": 0.0})


def test_window_ends_at_sunset_not_on_the_hour():
    fc = _forecast()
    wins = plan.windows(fc, [0.9] * len(fc), {"2026-10-12": "2026-10-12T18:17"}, now=pd.Timestamp("2026-10-12 06:00"))
    assert len(wins) == 1 and wins[0]["end"] == "18:17"


def test_thunderstorm_vetoes_even_if_taste_says_yes():
    fc = _forecast()
    fc.loc[fc["hour"] == 12, "thunder"] = True
    wins = plan.windows(fc, [0.9] * len(fc), now=pd.Timestamp("2026-10-12 06:00"))
    assert all(not (w["start"] <= "12:00" < w["end"]) for w in wins) and len(wins) == 2


def test_gemma_output_with_invented_numbers_is_rejected():
    b = {"Go": "tomorrow (Monday), 10:00–18:17", "Feels like": "26°C", "Shoes on at": "09:50",
         "Why it counts": "about 29 more days (a generic weather app would say about 9)"}
    assert nudge.grounded("Go tomorrow at 10:00, it feels like 26°C. About 29 days left. Shoes on at 09:50.", b)
    assert not nudge.grounded("It lasts 40 days. Shoes on at 09:50.", b)  # 40 is not a fact
    assert not nudge.grounded("Go tomorrow, 26°C. Shoes on soon.", b)  # missing the time
