import pandas as pd
import numpy as np
import joblib
import logging

from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ==========================================================
# CONFIG
# ==========================================================

INPUT_PATH = "data/processed/session_features.csv"
MODEL_PATH = "models/behavior_model.pkl"

TEST_SIZE = 0.2
RANDOM_STATE = 42

N_TREES = 300
BATCH_SIZE = 10

MIN_FEATURES = 5
MAX_FEATURES = 6  # 4–5 + mandatory ones


# ==========================================================
# LOGGING
# ==========================================================

logging.basicConfig(level=logging.INFO, format="🧠 %(message)s")
log = logging.getLogger()


# ==========================================================
# LOAD DATA
# ==========================================================

log.info("Loading dataset...")
df = pd.read_csv(INPUT_PATH)

log.info(f"Shape: {df.shape}")
log.info(f"Users: {df['PARTICIPANT_ID'].nunique()}")
log.info(f"Sessions: {df['TEST_SECTION_ID'].nunique()}")


# ==========================================================
# BASE FEATURES (CANDIDATES)
# ==========================================================

CANDIDATE_FEATURES = [
    "MEAN_IKI",
    "MEDIAN_IKI",
    "IKI_CV",
    "LONG_PAUSE_COUNT",
    "PAUSE_RATIO",
    "MEAN_HOLD_TIME",
    "MAX_HOLD_TIME",
    "TOTAL_KEYSTROKES",
    "BACKSPACE_COUNT",
    "BACKSPACE_RATIO",
    "MEAN_ERROR_RATE_ML",
    "MEAN_SENTENCE_LENGTH",
    "KSPC_PROXY",
]


MANDATORY_FEATURES = [
    "BACKSPACE_RATIO",
    "MEAN_ERROR_RATE_ML",
]


TARGETS = [
    "COGLOAD_PROXY",
    "STRESS_PROXY",
    "UNFOCUS_PROXY"
]


# ==========================================================
# CLEAN DATA
# ==========================================================

log.info("Cleaning data...")

df = df.replace([np.inf, -np.inf], np.nan)

missing_features = set(CANDIDATE_FEATURES + TARGETS) - set(df.columns)
if missing_features:
    raise ValueError(f"Missing columns: {missing_features}")

df = df.dropna(subset=TARGETS)


# ==========================================================
# CORRELATION FILTER (KEY PART)
# ==========================================================

def select_least_correlated_features(df, features, k=4):
    """
    Greedy selection:
    - start from least correlated feature set
    - ensure diversity
    """

    corr = df[features].corr().abs()

    selected = []

    # always include most important mandatory ones first
    for f in MANDATORY_FEATURES:
        if f in features:
            selected.append(f)

    remaining = [f for f in features if f not in selected]

    pbar = tqdm(total=k, desc="🧠 Selecting features")

    while len(selected) < k and remaining:

        best_feature = None
        best_score = -1

        for f in remaining:

            if len(selected) == 0:
                score = df[f].std()
            else:
                score = 1 - corr.loc[f, selected].mean()

            if score > best_score:
                best_score = score
                best_feature = f

        selected.append(best_feature)
        remaining.remove(best_feature)

        pbar.update(1)

    pbar.close()

    return selected


# ==========================================================
# FEATURE SELECTION
# ==========================================================

log.info("Selecting non-correlated features...")

SELECTED_FEATURES = select_least_correlated_features(
    df,
    CANDIDATE_FEATURES,
    k=MAX_FEATURES
)

log.info(f"Selected features: {SELECTED_FEATURES}")


# ==========================================================
# PREPARE X / Y
# ==========================================================

X = df[SELECTED_FEATURES].fillna(0)
y = df[TARGETS].copy()

# align safely
mask = ~y.isna().any(axis=1)
X = X.loc[mask]
y = y.loc[mask]

# normalize targets
y = (y - y.mean()) / (y.std() + 1e-9)


# ==========================================================
# TRAIN / TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

log.info(f"Train: {X_train.shape}, Test: {X_test.shape}")


# ==========================================================
# MODEL (warm-start + batch trees)
# ==========================================================

model = RandomForestRegressor(
    n_estimators=BATCH_SIZE,
    warm_start=True,
    max_depth=14,
    random_state=RANDOM_STATE,
    n_jobs=-1
)


# ==========================================================
# TRAINING WITH PROGRESS BAR
# ==========================================================

log.info("Training model...")

pbar = tqdm(total=N_TREES, desc="🌲 Trees")

for i in range(BATCH_SIZE, N_TREES + 1, BATCH_SIZE):

    model.set_params(n_estimators=i)
    model.fit(X_train, y_train)

    pbar.update(BATCH_SIZE)

pbar.close()


# ==========================================================
# EVALUATION
# ==========================================================

score = model.score(X_test, y_test)
log.info(f"R² score: {score:.4f}")


# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

importance = model.feature_importances_

importance_df = pd.DataFrame({
    "feature": SELECTED_FEATURES,
    "importance": importance
}).sort_values("importance", ascending=False)

log.info("Feature importance:")

total = importance_df["importance"].sum()

for _, row in importance_df.iterrows():
    log.info(f"{row['feature']:<25} → {100*row['importance']/total:.2f}%")


# ==========================================================
# SAVE MODEL
# ==========================================================

joblib.dump({
    "model": model,
    "features": SELECTED_FEATURES,
    "targets": TARGETS,
    "importance": importance_df
}, MODEL_PATH)

log.info(f"Saved → {MODEL_PATH}")
log.info("Done ✔")