import numpy as np
import joblib

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class BehaviorModel:
    """
    Population-level cognitive representation model.

    Learns latent cognitive state S_t from behavioral features X_t.
    """

    def __init__(self, n_components=3):
        self.scaler = StandardScaler()
        self.encoder = PCA(n_components=n_components)

    # ======================================================
    # TRAIN
    # ======================================================

    def fit(self, X):
        """
        X: behavioral features (from feature_engineering.py)
        """

        X_scaled = self.scaler.fit_transform(X)
        self.encoder.fit(X_scaled)

        return self

    # ======================================================
    # TRANSFORM
    # ======================================================

    def encode(self, X):
        """
        Returns latent cognitive state S_t
        """

        X_scaled = self.scaler.transform(X)
        S = self.encoder.transform(X_scaled)

        return S

    # ======================================================
    # SAVE / LOAD
    # ======================================================

    def save(self, path: str):
        joblib.dump({
            "scaler": self.scaler,
            "encoder": self.encoder
        }, path)

    def load(self, path: str):
        obj = joblib.load(path)
        self.scaler = obj["scaler"]
        self.encoder = obj["encoder"]
        return self