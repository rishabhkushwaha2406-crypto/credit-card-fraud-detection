"""Export everything the dashboard needs from the saved models.

Recreates the exact test split used in training, scores it with every saved model and writes
web/data/dashboard.json:
  - metrics at the default 0.5 threshold (from reports/results.json)
  - confusion counts at 99 thresholds per model (live threshold tuner + cost calculator)
  - ROC (log-spaced FPR grid) and interpolated precision-recall curves
  - the logistic-regression scorer (scaler + coefficients) and real test-set transactions

Run:  python -m src.export      (train.py calls this automatically)
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split

from src.data import add_features, clean, load_data

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
SCORER = "logistic_regression_undersampling"
THRESHOLDS = np.round(np.arange(0.01, 1.0, 0.01), 2)


def slug(model: str, strategy: str) -> str:
    return f"{model}_{strategy}".lower().replace(" ", "_")


def roc_points(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    grid = np.unique(np.concatenate([[0], np.logspace(-4, 0, 80)]))
    # best TPR achievable at or below each FPR (step interpolation)
    tpr_at = [float(tpr[fpr <= g].max()) for g in grid]
    return {"x": np.round(grid, 6).tolist(), "y": np.round(tpr_at, 4).tolist()}


def pr_points(y, p):
    prec, rec, _ = precision_recall_curve(y, p)
    grid = np.linspace(0, 1, 101)
    # interpolated precision: best precision at recall >= r
    prec_at = [float(prec[rec >= r].max()) if (rec >= r).any() else 0.0 for r in grid]
    return {"x": np.round(grid, 3).tolist(), "y": np.round(prec_at, 4).tolist()}


def main():
    df, source = load_data()
    df = add_features(clean(df))
    features = [c for c in df.columns if c != "Class"]
    X, y = df[features], df["Class"].values
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    amount = X_te["Amount"].values

    report = json.loads((ROOT / "reports" / "results.json").read_text())
    curves, sweep = {}, {}
    for r in report["results"]:
        key = f"{r['model']} | {r['strategy']}"
        pipe = joblib.load(ROOT / "models" / f"{slug(r['model'], r['strategy'])}.joblib")
        p = pipe.predict_proba(X_te)[:, 1]
        curves[key] = {"roc": roc_points(y_te, p), "pr": pr_points(y_te, p)}
        rows = []
        for t in THRESHOLDS:
            pred = p >= t
            tp = int((pred & (y_te == 1)).sum()); fp = int((pred & (y_te == 0)).sum())
            fn_mask = ~pred & (y_te == 1)
            rows.append([tp, fp, int(fn_mask.sum()), int((~pred & (y_te == 0)).sum()),
                         round(float(amount[fn_mask].sum()), 2)])
        sweep[key] = rows  # [tp, fp, fn, tn, missed_fraud_amount] per threshold

    # Scorer: logistic regression + undersampling (best-calibrated linear model)
    pipe = joblib.load(ROOT / "models" / f"{SCORER}.joblib")
    sc, clf = pipe.named_steps["scale"], pipe.named_steps["clf"]
    p_scorer = pipe.predict_proba(X_te)[:, 1]
    te = X_te.assign(Class=y_te, P=p_scorer)
    fraud = te[te.Class == 1]
    legit = te[te.Class == 0].sample(150, random_state=7)
    # include the legit transactions the model finds most suspicious, so false alarms are explorable
    legit = pd.concat([legit, te[te.Class == 0].nlargest(5, "P")]).drop_duplicates()
    pool = pd.concat([fraud, legit]).round(4)
    lo = X_te.quantile(0.005).round(2); hi = X_te.quantile(0.995).round(2)
    scorer = {
        "name": "Logistic Regression + Undersampling",
        "features": features,
        "mean": sc.mean_.round(6).tolist(), "scale": sc.scale_.round(6).tolist(),
        "coef": clf.coef_[0].round(6).tolist(), "intercept": round(float(clf.intercept_[0]), 6),
        "range": {f: [float(min(lo[f], fraud[f].min())), float(max(hi[f], fraud[f].max()))] for f in features},
        "pool": {
            "X": pool[features].values.tolist(),
            "y": pool["Class"].astype(int).tolist(),
        },
    }

    out = {**report, "thresholds": THRESHOLDS.tolist(), "sweep": sweep, "curves": curves,
           "scorer": scorer, "test_fraud_amount": round(float(amount[y_te == 1].sum()), 2)}
    (ROOT / "web" / "data").mkdir(parents=True, exist_ok=True)
    (ROOT / "web" / "data" / "dashboard.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"Exported web/data/dashboard.json ({len(sweep)} models x {len(THRESHOLDS)} thresholds, "
          f"{len(pool)} scorer transactions)")


if __name__ == "__main__":
    main()
