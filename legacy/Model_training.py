import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# ==========================================================
# LOAD DATA
# ==========================================================
df = pd.read_csv("session_cogload_metrics.csv")
df = df.sort_values(["PARTICIPANT_ID", "TEST_SECTION_ID"])

# ==========================================================
# 🔥 IMPROVED TEMPORAL TARGETS (LESS NOISY THAN ERROR RATE)
# ==========================================================
df["NEXT_IKI_STD"] = df.groupby("PARTICIPANT_ID")["STD_IKI"].shift(-1)
df["NEXT_PAUSE_RATIO"] = df.groupby("PARTICIPANT_ID")["PAUSE_RATIO"].shift(-1)

# NEW: stability signal instead of raw error prediction
df["IKI_STABILITY"] = 1 / (df["STD_IKI"] + 1e-6)

df["NEXT_IKI_STABILITY"] = df.groupby("PARTICIPANT_ID")["IKI_STABILITY"].shift(-1)

# ==========================================================
# CLEAN
# ==========================================================
df = df.dropna(subset=[
    "NEXT_IKI_STD",
    "NEXT_PAUSE_RATIO",
    "NEXT_IKI_STABILITY"
])

# ==========================================================
# 🧠 FEATURES (NO LEAKAGE + STABILITY FOCUS)
# ==========================================================
features = [
    # timing
    "MEDIAN_IKI",
    "STD_IKI",
    "IKI_CV",

    # behavior
    "TOTAL_KEYSTROKES",
    "BACKSPACE_COUNT",

    # session context
    "ELAPSED_TIME",
    "MEAN_SENTENCE_LENGTH",

    # cognitive indicators
    "PAUSE_RATIO",
    "MEAN_HOLD_TIME",

    # NEW: derived stability feature
    "IKI_STABILITY"
]

X = df[features]

Y = df[
    [
        "NEXT_IKI_STD",
        "NEXT_PAUSE_RATIO",
        "NEXT_IKI_STABILITY"
    ]
]

# ==========================================================
# 🧠 TIME-AWARE SPLIT (IMPORTANT FOR RL COMPATIBILITY)
# ==========================================================
split = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split], X.iloc[split:]
Y_train, Y_test = Y.iloc[:split], Y.iloc[split:]

# ==========================================================
# SCALE (NO LEAKAGE)
# ==========================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================================
# MODEL
# ==========================================================
base_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    max_depth=12
)

model = MultiOutputRegressor(base_model)
model.fit(X_train_scaled, Y_train)

# ==========================================================
# EVALUATION
# ==========================================================
pred = model.predict(X_test_scaled)

print("\n📊 IMPROVED RL-READY MODEL PERFORMANCE\n")

targets = Y.columns

for i, col in enumerate(targets):
    mse = mean_squared_error(Y_test.iloc[:, i], pred[:, i])
    r2 = r2_score(Y_test.iloc[:, i], pred[:, i])
    print(f"{col}: MSE={mse:.4f}, R2={r2:.4f}")

# ==========================================================
# SAVE MODEL
# ==========================================================
joblib.dump(model, "behavior_dynamics_model.pkl")
joblib.dump(scaler, "behavior_scaler.pkl")

print("\n✅ RL-READY MODEL SAVED")