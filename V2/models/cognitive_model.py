import numpy as np
import torch
from pathlib import Path
from collections import deque

from V2.models.behavior_net import BehaviorNet


class CognitiveModel:
    """
    Multi-output cognitive state estimator:
    - cognitive_load
    - stress
    - unfocus

    Uses sliding window + regression model (MSE trained).
    """

    def __init__(self, user_profile, model_path):

        self.profile = user_profile
        self.model_path = Path(model_path)

        # MUST match training script
        self.feature_order = [
            "BACKSPACE_RATIO",
            "MEAN_ERROR_RATE_ML",
            "MEAN_HOLD_TIME",
            "MEAN_SENTENCE_LENGTH",
            "PAUSE_RATIO",
            "IKI_CV",
        ]

        # -------------------------
        # Sliding window
        # -------------------------
        self.window_size = 20
        self.feature_window = deque(maxlen=self.window_size)

        # -------------------------
        # Load model (PyTorch only)
        # -------------------------
        checkpoint = torch.load(self.model_path, map_location="cpu")

        input_size = len(self.feature_order)
        output_size = 3  # (COGLOAD, STRESS, UNFOCUS)

        self.model = BehaviorNet(input_size, output_size)

        # load weights safely
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.targets = checkpoint.get(
            "targets",
            ["COGLOAD_PROXY", "STRESS_PROXY", "UNFOCUS_PROXY"]
        )

    # ----------------------------------
    # Normalize per-user
    # ----------------------------------
    def normalize(self, features):

        norm = {}

        for k, v in features.items():
            mean = self.profile.get_mean(k)
            std = self.profile.get_std(k)
            norm[k] = (v - mean) / (std + 1e-9)

        return norm

    # ----------------------------------
    # Vector builder
    # ----------------------------------
    def build_vector(self, norm_features):

        return np.array(
            [norm_features[f] for f in self.feature_order],
            dtype=np.float32
        ).reshape(1, -1)

    # ----------------------------------
    # Sliding window aggregation
    # ----------------------------------
    def aggregate_window(self):

        agg = {}

        for k in self.feature_order:
            vals = [f[k] for f in self.feature_window]
            agg[k] = float(np.mean(vals))

        return agg

    # ----------------------------------
    # Model inference
    # ----------------------------------
    def model_predict(self, X):

        with torch.no_grad():

            X_t = torch.tensor(X, dtype=torch.float32)

            output = self.model(X_t).cpu().numpy()[0]

            # raw regression outputs:
            # [cogload, stress, unfocus]
            return output

    # ----------------------------------
    # Public API
    # ----------------------------------
    def predict(self, feature_df):

        features = feature_df.iloc[0].to_dict()

        # add to sliding window
        self.feature_window.append(features)

        # warm-up phase
        if len(self.feature_window) < self.window_size:
            return {
                "cognitive_load": 0.0,
                "stress": 0.0,
                "unfocus": 0.0,
                "ready": False
            }

        # aggregate window
        window_features = self.aggregate_window()

        # normalize
        norm = self.normalize(window_features)

        # vectorize
        X = self.build_vector(norm)

        # predict
        cogload, stress, unfocus = self.model_predict(X)

        return {
            "cognitive_load": float(np.clip(cogload, 0, 1)),
            "stress": float(np.clip(stress, 0, 1)),
            "unfocus": float(np.clip(unfocus, 0, 1)),

            # optional debug (VERY useful for UI tuning)
            "raw": {
                "cogload": float(cogload),
                "stress": float(stress),
                "unfocus": float(unfocus),
            },

            "ready": True
        }