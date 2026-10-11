"""Learn one person's "would I actually go outside?" taste from ~60 labelled weather hours, with TabPFN.

TabPFN is a pretrained tabular foundation model: it does in-context learning on the labelled rows, so
there is no training loop and 60 examples is enough. Everything here runs on the local machine.
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tgw import DATA

LABELS = DATA / "labels.csv"
FEATURES = ["feels_c", "temp_c", "rain_mm", "wind_kmh", "cloud_pct", "humidity_pct", "hour"]

warnings.filterwarnings("ignore", category=UserWarning)


def make_tabpfn():
    # TabPFN v2: open weights on Hugging Face, no account or license login needed, runs fine on CPU.
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    return TabPFNClassifier.create_default_for_version(ModelVersion.V2, device="cpu", random_state=0)


def cards(history: pd.DataFrame, n: int = 60, seed: int = 7) -> pd.DataFrame:
    """Pick n daytime hours from history that cover the whole range of weather a year throws at you."""
    day = history[(history["hour"].between(8, 18)) & (history["is_day"] == 1)].copy()
    day["feels_bin"] = pd.cut(day["feels_c"], [-60, -5, 3, 10, 16, 22, 28, 60])
    day["wet"] = day["rain_mm"] >= 0.3
    rng = np.random.default_rng(seed)
    bins = day.groupby(["feels_bin", "wet"], observed=True)
    per_bin = max(1, n // bins.ngroups)
    picks = [g.sample(min(len(g), per_bin), random_state=int(rng.integers(1e9))) for _, g in bins]
    out = pd.concat(picks)
    if len(out) < n:
        out = pd.concat([out, day.drop(out.index).sample(n - len(out), random_state=seed)])
    return out.sample(frac=1, random_state=seed).head(n).reset_index(drop=True)


def load_labels() -> pd.DataFrame:
    return pd.read_csv(LABELS, parse_dates=["time"]) if LABELS.exists() else pd.DataFrame()


def weather_app_rule(df: pd.DataFrame) -> np.ndarray:
    """The generic 'nice weather' rule most apps use. This is the baseline to beat."""
    return (df["feels_c"].between(15, 27) & (df["rain_mm"] < 0.1) & (df["wind_kmh"] < 25)).astype(int).to_numpy()


def evaluate(labels: pd.DataFrame) -> dict:
    """Repeated 5-fold CV on the person's own labels: TabPFN vs logistic regression vs the generic rule."""
    X, y = labels[FEATURES].to_numpy(float), labels["go"].to_numpy(int)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
    scores = {k: {"acc": [], "auc": []} for k in ("weather_app_rule", "logistic_regression", "tabpfn")}
    for tr, te in cv.split(X, y):
        rule = weather_app_rule(labels.iloc[te])
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[tr], y[tr])
        tp = make_tabpfn().fit(X[tr], y[tr])
        for name, proba in (
            ("weather_app_rule", rule.astype(float)),
            ("logistic_regression", lr.predict_proba(X[te])[:, 1]),
            ("tabpfn", tp.predict_proba(X[te])[:, 1]),
        ):
            scores[name]["acc"].append(accuracy_score(y[te], proba >= 0.5))
            scores[name]["auc"].append(roc_auc_score(y[te], proba))
    out = {
        name: {m: round(float(np.mean(v)), 3) for m, v in s.items()} for name, s in scores.items()
    }
    out["n_labels"], out["n_yes"] = int(len(y)), int(y.sum())
    (DATA / "eval.json").write_text(json.dumps(out, indent=2))
    return out


class Taste:
    """Fitted TabPFN on all of the person's labels."""

    def __init__(self, labels: pd.DataFrame):
        self.model = make_tabpfn().fit(labels[FEATURES].to_numpy(float), labels["go"].to_numpy(int))

    def p_go(self, df: pd.DataFrame, batch: int = 2000) -> np.ndarray:
        X = df[FEATURES].to_numpy(float)
        return np.concatenate([self.model.predict_proba(X[i : i + batch])[:, 1] for i in range(0, len(X), batch)])
