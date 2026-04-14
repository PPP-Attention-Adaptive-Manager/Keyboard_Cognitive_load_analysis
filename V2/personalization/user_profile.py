# user_personalisation/user_profile.py

import numpy as np
import pandas as pd
import time
import json
import os


# ==========================================================
# USER PROFILE STORAGE
# ==========================================================

class UserProfileStore:
    """
    Maintains evolving user behavior statistics over time.
    Acts as long-term memory for RL personalization.
    """

    def __init__(self, path="user_personalisation/profiles.json"):
        self.path = path
        self.profiles = self._load()

    # ----------------------------------------------------------
    # LOAD / SAVE
    # ----------------------------------------------------------

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                return json.load(f)
        return {}

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.profiles, f, indent=2)

    # ----------------------------------------------------------
    # INIT USER
    # ----------------------------------------------------------

    def init_user(self, user_id):
        if user_id not in self.profiles:
            self.profiles[user_id] = {
                "sessions": 0,
                "mean_IKI": 0.0,
                "mean_pause": 0.0,
                "mean_backspace": 0.0,
                "var_IKI": 0.0,
                "last_timestamp": time.time()
            }

    # ----------------------------------------------------------
    # UPDATE USER PROFILE (TIME-AWARE)
    # ----------------------------------------------------------

    def update(self, user_id, session_features):
        """
        session_features = one row (aggregated session)
        """

        self.init_user(user_id)
        profile = self.profiles[user_id]

        profile["sessions"] += 1
        n = profile["sessions"]

        # learning rate decreases over time
        alpha = 1.0 / n

        # time delta (for decay)
        now = time.time()
        dt = now - profile["last_timestamp"]
        profile["last_timestamp"] = now
        
        # ------------------------------------------------------
        # exponential decay factor (more weight to recent sessions)
        # ------------------------------------------------------

        decay = np.exp(-dt / 60)  # 1-minute decay scale


        # ------------------------------------------------------
        # incremental updates
        # ------------------------------------------------------

        profile["mean_IKI"] = (
            (1 - alpha) * profile["mean_IKI"]
            + alpha * session_features["MEAN_IKI"]
        ) * decay + profile["mean_IKI"] * (1 - decay)

        profile["mean_pause"] = (
            (1 - alpha) * profile["mean_pause"]
            + alpha * session_features["PAUSE_RATIO"]
        )

        profile["mean_backspace"] = (
            (1 - alpha) * profile["mean_backspace"]
            + alpha * session_features["BACKSPACE_RATIO"]
        )

        # variance approximation
        ik = session_features["MEAN_IKI"]
        profile["var_IKI"] = (
            (1 - alpha) * profile["var_IKI"]
            + alpha * ((ik - profile["mean_IKI"]) ** 2)
        )

    # ----------------------------------------------------------
    # GET FEATURE VECTOR
    # ----------------------------------------------------------

    def get_vector(self, user_id):
        self.init_user(user_id)
        p = self.profiles[user_id]

        return np.array([
            p["mean_IKI"],
            p["mean_pause"],
            p["mean_backspace"],
            p["var_IKI"],
            p["sessions"]
        ], dtype=np.float32)