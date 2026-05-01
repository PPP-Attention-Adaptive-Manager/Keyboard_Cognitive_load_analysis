import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# =====================================================
# CONFIG
# =====================================================

DATA_PATH = Path(
    r"D:\files\PPP\github repo (linux)\Keyboard_Cognitive_load_analysis\V3\processed\phase1_final.parquet"
)

EMBED_DIM = 64
BATCH_SIZE = 16
EPOCHS = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================
# LOAD DATA
# =====================================================

def load_data():
    df = pd.read_parquet(DATA_PATH)

    if isinstance(df.iloc[0]["sequence"], str):
        df["sequence"] = df["sequence"].apply(json.loads)

    return df.reset_index(drop=True)


# =====================================================
# EXTRACT ENGINEERED FEATURES
# =====================================================

def extract_features(row):

    feature_cols = [c for c in row.index if c.startswith("feat_")]

    return np.array([row[c] for c in feature_cols], dtype=np.float32)


# =====================================================
# NORMALIZE SEQUENCE
# =====================================================

def normalize_sequence(seq):

    holds = np.array([s["hold"] for s in seq], dtype=np.float32)
    ikl   = np.array([s["ikl"] for s in seq], dtype=np.float32)
    code  = np.array([s["code"] for s in seq], dtype=np.float32)

    holds = np.nan_to_num(holds)
    ikl   = np.nan_to_num(ikl)

    holds = np.log1p(np.clip(holds, 0, 10000))
    ikl   = np.log1p(np.clip(ikl, 0, 20000))

    def safe_norm(x):
        std = x.std()
        if std < 1e-6:
            return x * 0.0
        return (x - x.mean()) / std

    holds = safe_norm(holds)
    ikl   = safe_norm(ikl)

    code = code / 255.0

    return np.stack([holds, ikl, code], axis=1)


# =====================================================
# DATASET
# =====================================================

class KeystrokeDataset(Dataset):

    def __init__(self, df):

        self.seq = df["sequence"].values
        self.feat = df[[c for c in df.columns if c.startswith("feat_")]].values

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):

        x_seq = normalize_sequence(self.seq[idx])
        x_feat = self.feat[idx].astype(np.float32)

        return (
            torch.tensor(x_seq, dtype=torch.float32),
            torch.tensor(x_feat, dtype=torch.float32),
        )


# =====================================================
# COLLATE FUNCTION
# =====================================================

def collate_fn(batch):

    seqs, feats = zip(*batch)

    lengths = torch.tensor([len(x) for x in seqs])

    max_len = max(lengths)

    padded_seq = torch.zeros(len(seqs), max_len, 3)

    for i, x in enumerate(seqs):
        padded_seq[i, :len(x)] = x

    feats = torch.stack(feats)

    return padded_seq, feats, lengths


# =====================================================
# MODEL (SEQUENCE + FEATURES → EMBEDDING)
# =====================================================

class KeystrokeEmbeddingModel(nn.Module):

    def __init__(self, feature_dim, embed_dim=64):

        super().__init__()

        # sequence encoder
        self.lstm = nn.LSTM(
            input_size=3,
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )

        self.seq_proj = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )

        # feature encoder
        self.feature_net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )

        # fusion
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x_seq, x_feat, lengths):

        packed = nn.utils.rnn.pack_padded_sequence(
            x_seq,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, (h, _) = self.lstm(packed)

        h = torch.cat([h[-2], h[-1]], dim=1)
        z_seq = self.seq_proj(h)

        z_feat = self.feature_net(x_feat)

        z = torch.cat([z_seq, z_feat], dim=1)

        return self.fusion(z)


# =====================================================
# CONTRASTIVE LOSS
# =====================================================

def contrastive_loss(z):

    z = F.normalize(z, dim=1)

    sim = torch.matmul(z, z.T)

    labels = torch.arange(len(z), device=z.device)

    return F.cross_entropy(sim, labels)


# =====================================================
# METRICS
# =====================================================

def contrastive_accuracy(z):

    z = F.normalize(z, dim=1)

    sim = torch.matmul(z, z.T)

    preds = sim.argmax(dim=1)

    labels = torch.arange(len(z), device=z.device)

    return (preds == labels).float().mean().item()


# =====================================================
# TRAIN LOOP
# =====================================================

def train(model, loader):

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    model.train()

    for epoch in range(EPOCHS):

        total_loss = 0
        total_acc = 0

        for x_seq, x_feat, lengths in tqdm(loader, desc=f"Epoch {epoch}"):

            x_seq = x_seq.to(DEVICE)
            x_feat = x_feat.to(DEVICE)

            z = model(x_seq, x_feat, lengths)

            loss = contrastive_loss(z)
            acc = contrastive_accuracy(z)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += acc

        print("\n====================")
        print(f"Epoch {epoch}")
        print(f"Loss: {total_loss:.4f}")
        print(f"Contrastive Acc: {total_acc / len(loader):.4f}")
        print("====================\n")


# =====================================================
# EXTRACT EMBEDDINGS
# =====================================================

def extract(model, loader):

    model.eval()
    out = []

    with torch.no_grad():

        for x_seq, x_feat, lengths in loader:

            x_seq = x_seq.to(DEVICE)
            x_feat = x_feat.to(DEVICE)

            z = model(x_seq, x_feat, lengths)

            out.append(z.cpu().numpy())

    return np.vstack(out)


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    print("Device:", DEVICE)

    df = load_data()

    # =========================
    # TRAIN / TEST SPLIT
    # =========================

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_ds = KeystrokeDataset(train_df)
    test_ds  = KeystrokeDataset(test_df)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = KeystrokeEmbeddingModel(
        feature_dim=train_df.filter(like="feat_").shape[1]
    ).to(DEVICE)

    # =========================
    # TRAIN
    # =========================
    print("Training...")
    train(model, train_loader)

    # =========================
    # EMBEDDINGS
    # =========================
    print("Extracting embeddings...")

    train_emb = extract(model, train_loader)
    test_emb  = extract(model, test_loader)

    np.save("train_embeddings.npy", train_emb)
    np.save("test_embeddings.npy", test_emb)

    print("\nDone!")
    print("Train:", train_emb.shape)
    print("Test:", test_emb.shape)