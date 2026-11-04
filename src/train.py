"""Train and compare fraud-detection models.

Models: Logistic Regression, Random Forest, XGBoost
Imbalance strategies: class weights, SMOTE, random undersampling
Metrics: precision, recall, F1, ROC-AUC, PR-AUC (average precision)

Run:  python -m src.train
Outputs: models/*.joblib, reports/results.json, web/data/dashboard.json (via src.export)
"""
import json
import time
from pathlib import Path

import joblib
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data import add_features, clean, load_data
from src.export import main as export_web

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def build_models(pos_weight: float):
    lr = lambda cw=None: LogisticRegression(max_iter=2000, class_weight=cw)
    rf = lambda cw=None: RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=2, n_jobs=-1,
        class_weight=cw, random_state=SEED)
    xgb = lambda spw=1.0: XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.08, subsample=0.9,
        colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="aucpr",
        n_jobs=-1, random_state=SEED)

    def pipe(clf, sampler=None):
        steps = [("scale", StandardScaler())]
        if sampler is not None:
            steps.append(("sample", sampler))
        steps.append(("clf", clf))
        return ImbPipeline(steps)

    smote = lambda: SMOTE(random_state=SEED)
    under = lambda: RandomUnderSampler(sampling_strategy=0.1, random_state=SEED)
    return {
        ("Logistic Regression", "Class weights"): pipe(lr("balanced")),
        ("Logistic Regression", "SMOTE"): pipe(lr(), smote()),
        ("Logistic Regression", "Undersampling"): pipe(lr(), under()),
        ("Random Forest", "Class weights"): pipe(rf("balanced_subsample")),
        ("Random Forest", "SMOTE"): pipe(rf(), smote()),
        ("Random Forest", "Undersampling"): pipe(rf(), under()),
        ("XGBoost", "Class weights"): pipe(xgb(pos_weight)),
        ("XGBoost", "SMOTE"): pipe(xgb(), smote()),
        ("XGBoost", "Undersampling"): pipe(xgb(), under()),
    }


def main():
    df, source = load_data()
    df = add_features(clean(df))
    features = [c for c in df.columns if c != "Class"]
    X, y = df[features], df["Class"].values
    print(f"Data source: {source} | rows={len(df):,} | fraud={y.sum():,} ({y.mean():.3%})")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    (ROOT / "models").mkdir(exist_ok=True)
    results = []
    for (model, strategy), pipe in build_models(pos_weight).items():
        t0 = time.time()
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
        row = {
            "model": model, "strategy": strategy,
            "precision": precision_score(y_te, pred, zero_division=0),
            "recall": recall_score(y_te, pred),
            "f1": f1_score(y_te, pred),
            "roc_auc": roc_auc_score(y_te, proba),
            "pr_auc": average_precision_score(y_te, proba),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "train_seconds": round(time.time() - t0, 1),
        }
        row = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
        results.append(row)
        print(f"{model:20s} {strategy:14s} P={row['precision']:.3f} R={row['recall']:.3f} "
              f"F1={row['f1']:.3f} ROC-AUC={row['roc_auc']:.4f}")

        slug = f"{model}_{strategy}".lower().replace(" ", "_")
        joblib.dump(pipe, ROOT / "models" / f"{slug}.joblib")

    best = max(results, key=lambda r: r["f1"])
    print(f"\nBest by F1: {best['model']} + {best['strategy']} (F1={best['f1']})")

    # Feature importance from the best tree model (XGBoost class-weighted)
    xgb_pipe = joblib.load(ROOT / "models" / "xgboost_class_weights.joblib")
    imp = sorted(zip(features, xgb_pipe.named_steps["clf"].feature_importances_), key=lambda t: -t[1])[:12]

    payload = {
        "data_source": source, "rows": int(len(df)), "fraud_count": int(y.sum()),
        "fraud_rate": round(float(y.mean()), 5), "test_size": int(len(y_te)),
        "results": results, "best": best,
        "feature_importance": [{"feature": f, "importance": round(float(v), 4)} for f, v in imp],
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "results.json").write_text(json.dumps(payload, indent=2))
    print("Saved reports/results.json")
    export_web()

if __name__ == "__main__":
    main()
