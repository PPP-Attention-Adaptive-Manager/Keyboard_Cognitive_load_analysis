# ============================================================
# 🧠 ROBUST KEYSTROKE MODEL TRAINING (NaN-SAFE + GROUP SAFE)
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import math

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# CONFIG
# ============================================================

V2_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = V2_DIR / "fully_labeled_cognitive_dataset.csv"

GROUP_COL = "PARTICIPANT_ID"
LABEL_COL = "cognitive_label"
RISK_COL = "cognitive_risk"

BASE_FEATURES = [
    "IKT",
    "dwell_time",
    "typing_speed",
    "burstiness",
    "ikt_mean_5",
    "dwell_mean_5",
    "dwell_std_5",
]

OPTIONAL_FEATURES = [
    "is_alpha",
    "is_digit",
    "is_space",
    "is_punct",
    "is_backspace",
    "keycode_norm",
    "activity_density",
]


# ============================================================
# FEATURE BUILDER (SAFE)
# ============================================================

def build_features(df):
    feats = []

    for c in BASE_FEATURES:
        if c in df.columns:
            feats.append(c)

    for c in OPTIONAL_FEATURES:
        if c in df.columns:
            feats.append(c)

    if not feats:
        raise ValueError("No usable features found!")

    print(f"🧠 Using {len(feats)} features")

    X = df[feats].apply(pd.to_numeric, errors="coerce")

    return X, feats


# ============================================================
# LOAD + CLEAN DATA (IMPORTANT FIX HERE)
# ============================================================

def load_data(task="both"):
    df = pd.read_csv(DATA_PATH, low_memory=False)

    df[GROUP_COL] = df[GROUP_COL].astype(str)

    # -----------------------------
    # CRITICAL FIX: REMOVE NaNs
    # -----------------------------

    required_cols = [GROUP_COL]

    if task in ["label", "both"]:
        required_cols.append(LABEL_COL)

    if task in ["risk", "both"]:
        required_cols.append(RISK_COL)

    df = df.dropna(subset=required_cols)

    # remove rows where features are completely missing
    df = df.dropna(how="all")

    X, features = build_features(df)

    # final cleanup: align indices after feature conversion
    valid_mask = ~X.isna().all(axis=1)
    df = df.loc[valid_mask]
    X = X.loc[valid_mask]

    groups = df[GROUP_COL]

    y_label = df[LABEL_COL] if LABEL_COL in df.columns else None
    y_risk = df[RISK_COL] if RISK_COL in df.columns else None

    return df, X, y_label, y_risk, groups, features


# ============================================================
# GROUP SPLIT (SAFE)
# ============================================================

def split(X, y, groups, test_size=0.2):

    # FINAL SAFETY CHECK (prevents sklearn crash)
    mask = ~pd.isna(y)
    X = X.loc[mask]
    y = y.loc[mask]
    groups = groups.loc[mask]

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    tr, te = next(gss.split(X, y, groups))

    return (
        X.iloc[tr].to_numpy(),
        X.iloc[te].to_numpy(),
        y.iloc[tr].to_numpy(),
        y.iloc[te].to_numpy()
    )


# ============================================================
# CLASSIFICATION
# ============================================================

def train_classifier(X_train, X_test, y_train, y_test):

    models = {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logreg": make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=1000)),
        "hgb": HistGradientBoostingClassifier()
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        results.append({
            "model": name,
            "acc": accuracy_score(y_test, pred),
            "bal_acc": balanced_accuracy_score(y_test, pred),
            "f1": f1_score(y_test, pred, average="macro")
        })

    return results


# ============================================================
# REGRESSION
# ============================================================

def train_regressor(X_train, X_test, y_train, y_test):

    models = {
        "dummy": DummyRegressor(strategy="mean"),
        "ridge": make_pipeline(SimpleImputer(), StandardScaler(), Ridge()),
        "hgb": HistGradientBoostingRegressor()
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        rmse = math.sqrt(mean_squared_error(y_test, pred))

        results.append({
            "model": name,
            "mae": mean_absolute_error(y_test, pred),
            "rmse": rmse,
            "r2": r2_score(y_test, pred)
        })

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    print("📦 Loading dataset...")
    df, X, y_label, y_risk, groups, features = load_data("both")

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    print("\n🚀 Classification training...")

    X_train, X_test, y_train, y_test = split(X, y_label, groups)

    clf_results = train_classifier(X_train, X_test, y_train, y_test)

    print("\n📊 Classification results:")
    for r in clf_results:
        print(r)

    # ========================================================
    # REGRESSION
    # ========================================================

    print("\n🚀 Regression training...")

    X_train, X_test, y_train, y_test = split(X, y_risk, groups)

    reg_results = train_regressor(X_train, X_test, y_train, y_test)

    print("\n📊 Regression results:")
    for r in reg_results:
        print(r)


if __name__ == "__main__":
    main()