# ============================================================
# 🧠 MEMORY-SAFE STUDENT LABELER (FIXED + BATCHED + STREAMING)
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import torch
import torch.nn as nn
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

SEQ_LEN = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_PARTICIPANTS = 500
CHUNKSIZE = 100_000   # 🔥 critical fix

V2_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = V2_DIR / "data/raw/merged_first_sections.csv"
MODEL_PATH = V2_DIR / "student_model.pt"
OUTPUT_PATH = V2_DIR / "fully_labeled_cognitive_dataset.csv"

# ============================================================
# FEATURES (MUST MATCH MODEL = 7 FEATURES)
# ============================================================

FEATURES = [
    "IKT",
    "dwell_time",
    "typing_speed",
    "burstiness",
    "ikt_mean_5",
    "dwell_mean_5",
    "dwell_std_5",
]

# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(g):
    g = g.copy()

    g["IKT"] = g["PRESS_TIME"].diff().fillna(0).clip(lower=0)
    g["dwell_time"] = (g["RELEASE_TIME"] - g["PRESS_TIME"]).fillna(0).clip(lower=0)
    g["typing_speed"] = 1.0 / (g["IKT"] + 1e-6)

    g["burstiness"] = g["IKT"].rolling(5, min_periods=1).std().fillna(0)
    g["ikt_mean_5"] = g["IKT"].rolling(5, min_periods=1).mean().fillna(0)
    g["dwell_mean_5"] = g["dwell_time"].rolling(5, min_periods=1).mean().fillna(0)
    g["dwell_std_5"] = g["dwell_time"].rolling(5, min_periods=1).std().fillna(0)

    return g


# ============================================================
# MODEL
# ============================================================

class Student(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=7,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.proj = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        h = torch.cat([h[0], h[1]], dim=1)
        return self.proj(h)


# ============================================================
# LOAD MODEL
# ============================================================

print("📦 Loading model...")

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

student = Student().to(DEVICE)
student.load_state_dict(ckpt, strict=True)
student.eval()

print("✅ Model loaded")


# ============================================================
# STREAMING PARTICIPANTS (FIX MEMORY ISSUE)
# ============================================================

def iter_participants():

    buffer = pd.DataFrame()

    for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNKSIZE, low_memory=False):

        chunk["PARTICIPANT_ID"] = chunk["PARTICIPANT_ID"].astype(str)

        buffer = pd.concat([buffer, chunk], ignore_index=True)

        # split complete participants
        for pid, g in buffer.groupby("PARTICIPANT_ID"):

            if len(g) < 50:
                continue

            buffer = buffer[buffer["PARTICIPANT_ID"] != pid]

            yield pid, g.sort_values("PRESS_TIME")

            if len(buffer) == 0:
                break


# ============================================================
# LIMIT 500 USERS
# ============================================================

def limited():
    n = 0
    for pid, g in iter_participants():
        n += 1
        if n > MAX_PARTICIPANTS:
            break
        yield pid, g


# ============================================================
# KMEANS
# ============================================================

kmeans = MiniBatchKMeans(n_clusters=3, batch_size=4096, random_state=42)


# ============================================================
# PASS 1 — EMBEDDINGS (BATCHED)
# ============================================================

print("\n🚀 Pass 1: embeddings")

buffer = []

for pid, g in tqdm(limited()):

    g = engineer_features(g)

    if len(g) < SEQ_LEN:
        continue

    f = g[FEATURES].to_numpy(np.float32)

    for i in range(len(g) - SEQ_LEN + 1):

        seq = torch.tensor(f[i:i+SEQ_LEN], dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            z = student(seq).cpu().numpy()[0]

        buffer.append(z)

        if len(buffer) > 20000:
            kmeans.partial_fit(np.array(buffer, dtype=np.float32))
            buffer = []

if buffer:
    kmeans.partial_fit(np.array(buffer, dtype=np.float32))

print("✅ KMeans ready")


# ============================================================
# PASS 2 — STATS
# ============================================================

print("\n🚀 Pass 2: stats")

dists = []

for pid, g in tqdm(limited()):

    g = engineer_features(g)

    if len(g) < SEQ_LEN:
        continue

    f = g[FEATURES].to_numpy(np.float32)

    for i in range(len(g) - SEQ_LEN + 1):

        seq = torch.tensor(f[i:i+SEQ_LEN], dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            z = student(seq).cpu().numpy()[0]

        c = kmeans.predict(z.reshape(1, -1))[0]
        dist = np.linalg.norm(z - kmeans.cluster_centers_[c])

        dists.append(dist)

dists = np.array(dists)
mean, std = dists.mean(), dists.std()

print("✅ Stats ready")


# ============================================================
# PASS 3 — LABELING (STREAM SAFE)
# ============================================================

print("\n🚀 Pass 3: labeling")

first = True
counts = Counter()

for pid, g in tqdm(limited()):

    g = engineer_features(g)

    labels = np.full(len(g), np.nan)
    risks = np.full(len(g), np.nan)

    if len(g) >= SEQ_LEN:

        f = g[FEATURES].to_numpy(np.float32)

        for i in range(len(g) - SEQ_LEN + 1):

            seq = torch.tensor(f[i:i+SEQ_LEN], dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                z = student(seq).cpu().numpy()[0]

            c = kmeans.predict(z.reshape(1, -1))[0]
            dist = np.linalg.norm(z - kmeans.cluster_centers_[c])

            risk = np.clip((dist - mean) / (3 * std + 1e-8), 0, 1)

            labels[i + SEQ_LEN - 1] = c
            risks[i + SEQ_LEN - 1] = risk

    g["cognitive_label"] = pd.Series(labels).ffill().to_numpy()
    g["cognitive_risk"] = pd.Series(risks).ffill().to_numpy()

    counts.update(pd.Series(g["cognitive_label"]).dropna().astype(int))

    g.to_csv(OUTPUT_PATH, mode="w" if first else "a", header=first, index=False)
    first = False

print("\n📊 Distribution:")
print(dict(counts))