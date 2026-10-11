"""Touch Grass Window: a local web app. Label weather you'd go out in, then get this week's outdoor windows.

Run:  python app.py            (then open http://127.0.0.1:5050)
      python app.py --offline  (uses the cached forecast; no network at all)
"""
import json
import sys
from datetime import date, datetime

import pandas as pd
from flask import Flask, redirect, render_template_string, request, url_for

from tgw import nudge, plan, taste, weather

OFFLINE = "--offline" in sys.argv
app = Flask(__name__)
CARDS = taste.DATA / "cards.csv"
STATE: dict = {}


def f_of(c: float) -> int:
    return round(c * 9 / 5 + 32)


def describe(row) -> dict:
    t = pd.Timestamp(row["time"])
    part = "early" if t.day <= 10 else "mid" if t.day <= 20 else "late"
    cloud, rain, wind = row["cloud_pct"], row["rain_mm"], row["wind_kmh"]
    return {
        "when": f"{t.strftime('%A')}, {part} {t.strftime('%B')} · {t.strftime('%I %p').lstrip('0')}",
        "feels": f"{row['feels_c']:.0f}°C ({f_of(row['feels_c'])}°F)",
        "temp": f"{row['temp_c']:.0f}°C",
        "sky": "Sunny" if cloud < 25 else "Partly cloudy" if cloud < 70 else "Overcast",
        "rain": "Dry" if rain < 0.1 else f"Light rain ({rain:.1f} mm/h)" if rain < 2 else f"Rain ({rain:.1f} mm/h)",
        "wind": f"{'Calm' if wind < 10 else 'Breezy' if wind < 25 else 'Windy' if wind < 40 else 'Very windy'} ({wind:.0f} km/h)",
        "humidity": f"{row['humidity_pct']:.0f}%",
    }


def get_cards() -> pd.DataFrame:
    if not CARDS.exists():
        taste.cards(weather.history()).to_csv(CARDS, index=False)
    return pd.read_csv(CARDS, parse_dates=["time"])


def compute() -> dict:
    hist = weather.history()
    labels = taste.load_labels()
    model = taste.Taste(labels)
    fc, meta = weather.forecast(offline=OFFLINE)
    fc["p"] = model.p_go(fc)
    wins = plan.windows(fc, fc["p"])
    left_cache = taste.DATA / "left.json"
    key = f"{date.today()}:{len(labels)}"
    left = json.loads(left_cache.read_text()) if left_cache.exists() else {}
    if left.get("key") != key:
        left = {"key": key, **plan.good_days_left(hist, model)}
        left_cache.write_text(json.dumps(left))
    ev_file = taste.DATA / "eval.json"
    ev = json.loads(ev_file.read_text()) if ev_file.exists() else {}
    if ev.get("n_labels") != len(labels):
        ev = taste.evaluate(labels)
    today = date.today().isoformat()
    todays = [w for w in wins if w["date"] == today]
    best = max(todays or wins, key=lambda w: w["hours"] * w["p"]) if wins else None
    sunset = meta["daily"].get(today, (None, None))[1]
    facts = {
        "now": datetime.now().strftime("%a %b %d %H:%M"),
        "sunset_today": sunset[11:] if sunset else None,
        "best_window": best,
        "best_window_is_today": bool(todays),
        "other_windows_this_week": [w for w in wins if w is not best][:4],
        "good_days_left": {
            "meaning": "typical-year average number of days from tomorrow to Dec 31 with 2+ hours this person would go out in (2016-2025 history)",
            "average": left["average"],
            "range": [left["min"], left["max"]],
        },
    }
    msg, source = nudge.write(facts)
    grid = {}
    for _, r in fc[fc["hour"].between(7, 19)].iterrows():
        grid.setdefault(r["time"].strftime("%a %d"), []).append(
            {
                "h": int(r["hour"]),
                "p": round(float(r["p"]), 2),
                "day": int(r["is_day"]),
                "rain": int(r["rain_prob"]),
                "tip": f"{r['time']:%a %H:00} · feels {r['feels_c']:.0f}°C · wind {r['wind_kmh']:.0f} km/h · rain {int(r['rain_prob'])}% · p(go) {r['p']:.2f}",
            }
        )
    return {"msg": msg, "source": source, "wins": wins, "left": left, "ev": ev, "grid": grid,
            "fetched": meta["fetched"], "n": len(labels), "yes": int(labels["go"].sum()), "offline": OFFLINE}


@app.get("/")
def home():
    if len(taste.load_labels()) < 20:
        return redirect(url_for("label"))
    if "plan" not in STATE:
        STATE["plan"] = compute()
    return render_template_string(PLAN_HTML, **STATE["plan"])


@app.post("/refresh")
def refresh():
    STATE.pop("plan", None)
    return redirect(url_for("home"))


@app.route("/label", methods=["GET", "POST"])
def label():
    cards = get_cards()
    labels = taste.load_labels()
    if request.method == "POST":
        row = cards.iloc[int(request.form["i"])]
        if request.form["go"] != "skip":
            new = row[["time"] + taste.FEATURES].to_frame().T.assign(go=int(request.form["go"]))
            pd.concat([labels, new]).to_csv(taste.LABELS, index=False)
        skipped = STATE.setdefault("skipped", set())
        if request.form["go"] == "skip":
            skipped.add(int(request.form["i"]))
        STATE.pop("plan", None)
        return redirect(url_for("label"))
    done = set(labels["time"].astype(str)) if len(labels) else set()
    todo = [i for i, t in enumerate(cards["time"].astype(str)) if t not in done and i not in STATE.get("skipped", set())]
    if not todo:
        return redirect(url_for("home"))
    i = todo[0]
    return render_template_string(LABEL_HTML, i=i, c=describe(cards.iloc[i]), n=len(labels), total=len(cards))


BASE_CSS = """
:root{--bg:#f6f7f2;--card:#fff;--ink:#1d2a22;--mute:#5f6b63;--line:#dfe4dc;--go:#1f6f5f;--no:#9a4b3c}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:980px;margin:0 auto;padding:28px 16px 60px}h1{font-size:26px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 10px}
.sub{color:var(--mute);margin:0 0 20px}.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px}
.mute{color:var(--mute);font-size:13px}button{font:inherit;border:0;border-radius:10px;padding:12px 22px;cursor:pointer}
"""

LABEL_HTML = """<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Teach it your taste</title><style>""" + BASE_CSS + """
.big{font-size:44px;font-weight:700;margin:6px 0}.row{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:16px 0}
.row div{background:var(--bg);border-radius:10px;padding:10px}.row b{display:block;font-size:12px;color:var(--mute);font-weight:500}
.btns{display:flex;gap:10px;margin-top:18px}.yes{background:var(--go);color:#fff}.no{background:var(--no);color:#fff}.skip{background:var(--line)}
.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin:6px 0 20px}.bar i{display:block;height:100%;background:var(--go)}
@media(max-width:600px){.row{grid-template-columns:repeat(2,1fr)}}
</style></head><body><main>
<h1>Teach it your taste</h1><p class=sub>Real Champaign weather from the past. You have a free hour. Do you go outside for a walk?</p>
<div class=mute>{{n}} of {{total}} labelled · TabPFN needs about 40–60</div><div class=bar><i style="width:{{ (100*n/total)|round }}%"></i></div>
<div class=card><div class=mute>{{c.when}}</div><div class=big>Feels {{c.feels}}</div>
<div class=row><div><b>Sky</b>{{c.sky}}</div><div><b>Rain</b>{{c.rain}}</div><div><b>Wind</b>{{c.wind}}</div><div><b>Humidity</b>{{c.humidity}}</div></div>
<form method=post class=btns><input type=hidden name=i value={{i}}>
<button class=yes name=go value=1 id=y>Yes, I'd go (Y)</button><button class=no name=go value=0 id=n>No (N)</button><button class=skip name=go value=skip id=s>Skip (S)</button></form></div>
{% if n >= 20 %}<p><a href="/">Enough for now, show my week →</a></p>{% endif %}
<p class=mute>Your answers are saved to data/labels.csv on this computer and nowhere else.</p>
<script>document.onkeydown=e=>{const k={y:'y',n:'n',s:'s'}[e.key.toLowerCase()];if(k)document.getElementById(k).click()}</script>
</main></body></html>"""

PLAN_HTML = """<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Touch Grass Window</title><style>""" + BASE_CSS + """
.nudge{font-size:20px;line-height:1.45;border-left:5px solid var(--go)}.grid{display:grid;grid-template-columns:70px repeat(13,1fr);gap:3px;font-size:12px}
.cell{height:30px;border-radius:5px;background:#e9ece6}.night{background:repeating-linear-gradient(45deg,#e9ece6,#e9ece6 4px,#f6f7f2 4px,#f6f7f2 8px)}
.rain{outline:2px dashed #6b7c8f;outline-offset:-2px}.hdr{color:var(--mute);text-align:center}.dl{color:var(--mute);align-self:center}
table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line)}th{font-size:13px;color:var(--mute);font-weight:500}
.stat{font-size:40px;font-weight:700;color:var(--go)}.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}.win b{color:var(--go)}
@media(max-width:700px){.two{grid-template-columns:1fr}.grid{grid-template-columns:52px repeat(13,1fr)}}
</style></head><body><main>
<h1>Touch Grass Window</h1><p class=sub>Champaign, IL · forecast {{fetched}}{% if offline %} (offline, cached){% endif %} · taste learned from {{n}} of your answers</p>
<div class="card nudge">{{msg}}<div class=mute style="margin-top:10px">Written by {{source}} running locally via Ollama</div></div>
<h2>This week, hour by hour</h2><div class=card><div class=grid><div></div>{% for h in range(7,20) %}<div class=hdr>{{h}}</div>{% endfor %}
{% for d, cells in grid.items() %}<div class=dl>{{d}}</div>{% for c in cells %}<div class="cell{% if not c.day %} night{% endif %}{% if c.rain >= 60 %} rain{% endif %}" title="{{c.tip}}"
{% if c.day %}style="background:rgba(31,111,95,{{ '%.2f'|format(0.08 + 0.92*c.p) }})"{% endif %}></div>{% endfor %}{% endfor %}</div>
<p class=mute>Darker green = more likely you'd want to be out. Hatched = dark. Dashed = 60%+ chance of rain.</p></div>
<div class=two><div><h2>Your windows</h2><div class=card>{% for w in wins[:6] %}<div class=win><b>{{w.day}} {{w.start}}–{{w.end}}</b> · feels {{w.feels_c}}°C · p {{w.p}}</div>{% else %}No window clears your bar this week.{% endfor %}</div></div>
<div><h2>Days like this left this year</h2><div class=card><div class=stat>~{{left.average}}</div>
<div class=mute>Average number of days from tomorrow to Dec 31 with 2+ hours you'd go out in, using your taste on 2016–2025 Champaign weather (range {{left.min}}–{{left.max}}).</div></div></div></div>
<h2>Does it actually learn my taste?</h2><div class=card><table><tr><th>Model (5-fold CV ×3 on your {{ev.n_labels}} answers)</th><th>Accuracy</th><th>ROC AUC</th></tr>
<tr><td>Generic "nice weather" rule (feels 15–27°C, dry, wind &lt; 25)</td><td>{{ev.weather_app_rule.acc}}</td><td>{{ev.weather_app_rule.auc}}</td></tr>
<tr><td>Logistic regression</td><td>{{ev.logistic_regression.acc}}</td><td>{{ev.logistic_regression.auc}}</td></tr>
<tr><td><b>TabPFN (open weights, in-context, no training loop)</b></td><td><b>{{ev.tabpfn.acc}}</b></td><td><b>{{ev.tabpfn.auc}}</b></td></tr></table></div>
<form method=post action=/refresh style="margin-top:16px;display:flex;gap:10px"><button class=yes style="background:var(--go);color:#fff">Refresh</button><a href=/label style="align-self:center">Label more</a></form>
<p class=mute>Weather: Open-Meteo (CC BY 4.0). Your labels, schedule and this page stay on this computer.</p>
</main></body></html>"""

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
