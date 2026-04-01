import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import time
from tqdm import tqdm
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# Logging setup
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
def log(msg):
    logging.info(msg)

start_total = time.time()

# -----------------------------
# Load dataset
# -----------------------------
log("Loading dataset...")
df = pd.read_csv("session_cogload_metrics.csv")
log(f"Original shape: {df.shape}")

# -----------------------------
# Sample rows for large datasets
# -----------------------------
SAMPLE_SIZE = 20000
print("Dataframe length is" ,len(df))
if len(df) > SAMPLE_SIZE:
    df = df.sample(n=SAMPLE_SIZE, random_state=42)
log(f"Using shape: {df.shape}")

# -----------------------------
# Feature engineering
# -----------------------------
log("Engineering features...")
df['TOTAL_KEYSTROKES'] = df['TOTAL_KEYSTROKES'].replace(0, 1)
df['BACKSPACE_RATIO'] = df['BACKSPACE_COUNT'] / df['TOTAL_KEYSTROKES']
df['PAUSE_PER_MIN'] = df['LONG_PAUSE_COUNT'] / (df['ELAPSED_TIME'] + 1e-5)

# -----------------------------
# Complex proxy formula (independent from model features)
# -----------------------------
log("Creating complex cognitive load proxy...")
df['COGNITIVE_LOAD_PROXY'] = (
    0.3 * np.log1p(df['MEAN_HOLD_TIME'] * df['PAUSE_RATIO']) +
    0.2 * np.sqrt(df['MAX_HOLD_TIME']) +
    0.2 * (df['PAUSE_PER_MIN'] ** 1.5) +
    0.1 * np.tanh(df['LONG_PAUSE_COUNT']) +
    0.2 * np.log1p(df['MEAN_SENTENCE_LENGTH'] * (1 + df['BACKSPACE_RATIO']))
)

# -----------------------------
# Independent features for model
# -----------------------------
features = [
    'MEDIAN_IKI', 'STD_IKI', 'IKI_CV', 'TOTAL_KEYSTROKES',
    'ELAPSED_TIME', 'MEAN_SENTENCE_LENGTH', 'BACKSPACE_RATIO','MEAN_HOLD_TIME','PAUSE_RATIO','PAUSE_PER_MIN'
]
X = df[features]
y = df['COGNITIVE_LOAD_PROXY']
# Ensure there are no NaNs in target variable y
y = y.dropna()
X = X.loc[y.index]  # Sync X with y after dropping NaNs

# -----------------------------
# Scale features
# -----------------------------
log("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
log(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# -----------------------------
# Random Forest with warm_start for progress tracking
# -----------------------------
log("Training Random Forest and tracking progress...")
trees_range = list(range(10, 201, 10))
mse_progress = []

rf_temp = RandomForestRegressor(warm_start=True, n_jobs=-1, random_state=42)
for n_trees in tqdm(trees_range, desc="Training progress"):
    rf_temp.set_params(n_estimators=n_trees)
    rf_temp.fit(X_train, y_train)
    y_pred_temp = rf_temp.predict(X_test)
    mse_progress.append(mean_squared_error(y_test, y_pred_temp))

# -----------------------------
# Final model
# -----------------------------
rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
mse_final = mean_squared_error(y_test, y_pred)
r2_final = r2_score(y_test, y_pred)

print("\n📊 MODEL PERFORMANCE (Complex Proxy, Independent Features)")
print(f"MSE: {mse_final:.4f}")
print(f"R²:  {r2_final:.4f}")

# -----------------------------
# Feature importance
# -----------------------------
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print("\n🔥 Feature Importances:")
for f in tqdm(importances.index, desc="Ranking features"):
    pass
print(importances)

# -----------------------------
# Save model + scaler
# -----------------------------
log("Saving model and scaler...")
joblib.dump(rf, "rf_cogload_model_complex.pkl")
joblib.dump(scaler, "feature_scaler_complex.pkl")
log("✅ Model + scaler saved!")

# -----------------------------
# Visualizations
# -----------------------------
log("Visualizing training progress and results...")

# MSE vs Trees
plt.figure(figsize=(10,5))
plt.plot(trees_range, mse_progress, marker='o', color='dodgerblue')
plt.title("Random Forest Test MSE vs Number of Trees")
plt.xlabel("Number of Trees")
plt.ylabel("Test MSE")
plt.grid(True)
plt.show()

# Proxy distribution
plt.figure(figsize=(10,5))
sns.histplot(df['COGNITIVE_LOAD_PROXY'], bins=50, kde=True, color='skyblue')
plt.title("Distribution of Complex Cognitive Load Proxy")
plt.xlabel("Cognitive Load Proxy")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Predicted vs actual
plt.figure(figsize=(8,8))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Proxy")
plt.ylabel("Predicted Proxy")
plt.title("Predicted vs Actual Cognitive Load Proxy")
plt.tight_layout()
plt.show()

# Feature importance bar chart
plt.figure(figsize=(10,6))
sns.barplot(x=importances.values, y=importances.index, palette="viridis")
plt.title("Random Forest Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

log(f"Total runtime: {time.time() - start_total:.2f} sec")