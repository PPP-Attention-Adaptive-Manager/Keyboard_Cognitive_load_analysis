import pandas as pd
import numpy as np
import joblib
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ==========================================================
# CUDA / DEVICE CONFIG
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_gpu = f"🚀 Using Device: {device}"
if device.type == 'cuda':
    log_gpu += f" ({torch.cuda.get_device_name(0)})"

# ==========================================================
# CONFIG
# ==========================================================
INPUT_PATH = "data/processed/session_features.csv"
MODEL_PATH = "models/behavior_model.pth" # Changed extension for PyTorch

TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.001
MAX_FEATURES = 6

# ==========================================================
# LOGGING
# ==========================================================
logging.basicConfig(level=logging.INFO, format="🧠 %(message)s")
log = logging.getLogger()
log.info(log_gpu)

# ==========================================================
# LOAD & CLEAN DATA
# ==========================================================
log.info("Loading dataset...")
df = pd.read_csv(INPUT_PATH).replace([np.inf, -np.inf], np.nan)

CANDIDATE_FEATURES = [
    "MEAN_IKI", "MEDIAN_IKI", "IKI_CV", "LONG_PAUSE_COUNT",
    "PAUSE_RATIO", "MEAN_HOLD_TIME", "MAX_HOLD_TIME",
    "TOTAL_KEYSTROKES", "BACKSPACE_COUNT", "BACKSPACE_RATIO",
    "MEAN_ERROR_RATE_ML", "MEAN_SENTENCE_LENGTH", "KSPC_PROXY",
]
MANDATORY_FEATURES = ["BACKSPACE_RATIO", "MEAN_ERROR_RATE_ML"]
TARGETS = ["COGLOAD_PROXY", "STRESS_PROXY", "UNFOCUS_PROXY"]

df = df.dropna(subset=TARGETS)

# Feature Selection
def select_features(df, features, k=5):
    corr = df[features].corr().abs()
    selected = list(set(MANDATORY_FEATURES) & set(features))
    remaining = [f for f in features if f not in selected]
    while len(selected) < k and remaining:
        scores = {f: (1 - corr.loc[f, selected].mean() if selected else 1) for f in remaining}
        best_f = max(scores, key=scores.get)
        selected.append(best_f)
        remaining.remove(best_f)
    return selected

SELECTED_FEATURES = select_features(df, CANDIDATE_FEATURES, k=MAX_FEATURES)
log.info(f"Features: {SELECTED_FEATURES}")

# Normalize and Convert to Tensors
X = df[SELECTED_FEATURES].fillna(0).values.astype(np.float32)
y = df[TARGETS].values.astype(np.float32)

X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
y = (y - y.mean(axis=0)) / (y.std(axis=0) + 1e-9)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ==========================================================
# MODEL DEFINITION (PyTorch)
# ==========================================================
class BehaviorNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(BehaviorNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        return self.net(x)

model = BehaviorNet(len(SELECTED_FEATURES), len(TARGETS)).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==========================================================
# TRAINING LOOP
# ==========================================================
log.info("Starting PyTorch CUDA training...")
model.train()

pbar = tqdm(total=EPOCHS, desc="🔥 CUDA Epochs")
for epoch in range(EPOCHS):
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    pbar.update(1)
    pbar.set_postfix({"Loss": f"{epoch_loss/len(train_loader):.4f}"})
pbar.close()

# ==========================================================
# EVALUATION
# ==========================================================
model.eval()
with torch.no_grad():
    X_test_tensor = torch.from_numpy(X_test).to(device)
    y_test_tensor = torch.from_numpy(y_test).to(device)
    preds = model(X_test_tensor)
    test_loss = criterion(preds, y_test_tensor)
    log.info(f"Final Test MSE: {test_loss.item():.4f}")

# ==========================================================
# SAVE MODEL
# ==========================================================
# We save the state dict and the feature metadata
torch.save({
    'model_state_dict': model.state_dict(),
    'features': SELECTED_FEATURES,
    'targets': TARGETS
}, MODEL_PATH)

log.info(f"Saved model to {MODEL_PATH}")
log.info("Done ✔")