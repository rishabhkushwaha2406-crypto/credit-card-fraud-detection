"""Data loading.

If `data/creditcard.csv` (the Kaggle "Credit Card Fraud Detection" dataset by ULB,
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) is present, it is used.
Otherwise a synthetic dataset with the same schema (Time, V1..V28, Amount, Class)
and a similar fraud rate (~0.17%) is generated so the pipeline runs end to end.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "creditcard.csv"
FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def make_synthetic(n_samples: int = 100_000, fraud_rate: float = 0.0017, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=28,
        n_informative=10,
        n_redundant=6,
        n_clusters_per_class=1,
        weights=[1 - fraud_rate],
        flip_y=0.0005,
        class_sep=2.0,
        random_state=seed,
    )
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", np.sort(rng.uniform(0, 172_792, n_samples)).round())
    # Legit amounts: log-normal around ~$22; fraud skews to small "test" charges and large hits
    amount = rng.lognormal(3.1, 1.3, n_samples)
    fraud_idx = np.where(y == 1)[0]
    amount[fraud_idx] = np.where(
        rng.random(len(fraud_idx)) < 0.55,
        rng.uniform(0.5, 5, len(fraud_idx)),
        rng.lognormal(5.2, 0.9, len(fraud_idx)),
    )
    df["Amount"] = amount.round(2)
    df["Class"] = y
    return df


def load_data() -> tuple[pd.DataFrame, str]:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH), "kaggle"
    return make_synthetic(), "synthetic"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates and rows with missing values; enforce dtypes."""
    before = len(df)
    df = df.drop_duplicates().dropna()
    df["Class"] = df["Class"].astype(int)
    print(f"Cleaning: removed {before - len(df)} duplicate/missing rows")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Simple engineered features: log amount and hour of day."""
    df = df.copy()
    df["LogAmount"] = np.log1p(df["Amount"])
    df["Hour"] = (df["Time"] // 3600) % 24
    return df
