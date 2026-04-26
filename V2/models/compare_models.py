import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, r2_score

# =====================================================
# CONFIG
# =====================================================
DATA_PATH = "data/processed/session_features.csv"

NN_MODEL_PATH = "models/behavior_model.pth"
RF_MODEL_PATH = "models/behavior_model.pkl"

SUBSET_SIZE = 400
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\n🚀 Device: {DEVICE}")

# =====================================================
# PYTORCH MODEL ARCHITECTURE
# =====================================================
class BehaviorNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
        )

    def forward(self, x):
        return self.net(x)

# =====================================================
# LOAD NEURAL NETWORK
# =====================================================
print("\nLoading Neural Network...")

checkpoint = torch.load(NN_MODEL_PATH, map_location=DEVICE)

FEATURES = checkpoint["features"]
TARGETS = checkpoint["targets"]

nn_model = BehaviorNet(len(FEATURES), len(TARGETS))
nn_model.load_state_dict(checkpoint["model_state_dict"])
nn_model.to(DEVICE)
nn_model.eval()

# =====================================================
# LOAD RANDOM FOREST (ROBUST LOADER)
# =====================================================
print("Loading Random Forest...")

rf_checkpoint = joblib.load(RF_MODEL_PATH)

if hasattr(rf_checkpoint, "predict"):
    rf_model = rf_checkpoint

elif isinstance(rf_checkpoint, dict):
    rf_model = None
    for k, v in rf_checkpoint.items():
        if hasattr(v, "predict"):
            rf_model = v
            print(f"✅ RF model found under key: {k}")
            break

    if rf_model is None:
        raise ValueError("No sklearn model found inside .pkl")

else:
    raise ValueError("Unknown RandomForest format")

# =====================================================
# LOAD DATASET
# =====================================================
print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH).replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=TARGETS)

X = df[FEATURES].fillna(0).values.astype(np.float32)
y = df[TARGETS].values.astype(np.float32)

# =====================================================
# NORMALIZATION (same logic as training)
# =====================================================
X_mean, X_std = X.mean(axis=0), X.std(axis=0) + 1e-9
y_mean, y_std = y.mean(axis=0), y.std(axis=0) + 1e-9

X = (X - X_mean) / X_std
y = (y - y_mean) / y_std

# =====================================================
# SAME RANDOM SUBSET
# =====================================================
np.random.seed(42)
idx = np.random.choice(len(X), size=min(SUBSET_SIZE, len(X)), replace=False)

X_subset = X[idx]
y_subset = y[idx]

X_tensor = torch.tensor(X_subset).to(DEVICE)

# =====================================================
# PREDICTIONS
# =====================================================
print("\nRunning predictions...")

with torch.no_grad():
    nn_preds = nn_model(X_tensor).cpu().numpy()

rf_preds = rf_model.predict(X_subset)

# =====================================================
# MODEL PERFORMANCE
# =====================================================
def evaluate(name, preds):
    mse = mean_squared_error(y_subset, preds)
    r2 = r2_score(y_subset, preds)
    return {
        "Model": name,
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2,
    }

results = [
    evaluate("NeuralNetwork", nn_preds),
    evaluate("RandomForest", rf_preds),
]

comparison = pd.DataFrame(results).set_index("Model")

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")
print(comparison)

# =====================================================
# PER TARGET METRICS
# =====================================================
print("\nPer Target MSE")

for i, t in enumerate(TARGETS):
    nn_mse = mean_squared_error(y_subset[:, i], nn_preds[:, i])
    rf_mse = mean_squared_error(y_subset[:, i], rf_preds[:, i])

    print(f"{t}")
    print(f"  NN : {nn_mse:.4f}")
    print(f"  RF : {rf_mse:.4f}")

# =====================================================
# PERMUTATION IMPORTANCE (FAIR COMPARISON)
# =====================================================
print("\nComputing permutation importance...")

baseline_nn = mean_squared_error(y_subset, nn_preds)
baseline_rf = mean_squared_error(y_subset, rf_preds)

def perm_importance_nn():
    scores = {}
    for i, f in enumerate(FEATURES):

        Xp = X_subset.copy()
        np.random.shuffle(Xp[:, i])

        with torch.no_grad():
            preds = nn_model(torch.tensor(Xp).to(DEVICE)).cpu().numpy()

        scores[f] = mean_squared_error(y_subset, preds) - baseline_nn
    return scores


def perm_importance_rf():
    scores = {}
    for i, f in enumerate(FEATURES):

        Xp = X_subset.copy()
        np.random.shuffle(Xp[:, i])

        preds = rf_model.predict(Xp)
        scores[f] = mean_squared_error(y_subset, preds) - baseline_rf
    return scores

nn_perm = perm_importance_nn()
rf_perm = perm_importance_rf()

# =====================================================
# RF NATIVE IMPORTANCE
# =====================================================
if hasattr(rf_model, "feature_importances_"):
    rf_native = dict(zip(FEATURES, rf_model.feature_importances_))
else:
    rf_native = {f: 0 for f in FEATURES}

# =====================================================
# NN GRADIENT IMPORTANCE
# =====================================================
X_grad = torch.tensor(X_subset, requires_grad=True).to(DEVICE)
out = nn_model(X_grad).sum()
out.backward()

grad_imp = X_grad.grad.abs().mean(0).cpu().numpy()
nn_grad = dict(zip(FEATURES, grad_imp))

# =====================================================
# FINAL FEATURE COMPARISON
# =====================================================
importance_df = pd.DataFrame({
    "NN_Permutation": nn_perm,
    "RF_Permutation": rf_perm,
    "RF_Native": rf_native,
    "NN_Gradient": nn_grad,
})

importance_df = importance_df.sort_values(
    by="NN_Permutation",
    ascending=False
)

print("\n==============================")
print("FEATURE CONTRIBUTION ANALYSIS")
print("==============================")
print(importance_df)

print("\n✔ Comparison Complete")