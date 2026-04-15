import numpy as np


class CognitiveModel:
    """
    Non-RL cognitive load estimator.
    Deterministic + explainable.
    """

    def __init__(self, user_profile):

        self.profile = user_profile

        # weights learned manually (baseline)
        self.weights = {
            "BACKSPACE_RATIO": 1.2,
            "MEAN_ERROR_RATE_ML": 1.5,
            "MEAN_HOLD_TIME": 0.9,
            "MEAN_SENTENCE_LENGTH": -0.4,
            "PAUSE_RATIO": 1.3,
            "IKI_CV": 1.1,
        }

    # ----------------------------------
    # Normalization using user baseline
    # ----------------------------------
    def normalize(self, features):

        norm = {}

        for k, v in features.items():

            mean = self.profile.get_mean(k)
            std = self.profile.get_std(k)

            norm[k] = (v - mean) / (std + 1e-9)

        return norm

    # ----------------------------------
    # Cognitive Load Prediction
    # ----------------------------------
    def predict(self, feature_df):

        features = feature_df.iloc[0].to_dict()

        norm = self.normalize(features)

        score = 0

        for k, v in norm.items():
            score += self.weights[k] * v

        cogload = 1 / (1 + np.exp(-score))  # sigmoid

        return {
            "cognitive_load": float(cogload),
            "focus": float(1 - cogload),
            "fatigue": float(max(0, cogload - 0.5) * 2),
            "stress": float(min(1, cogload * 1.2)),
        }