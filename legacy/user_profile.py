import numpy as np
import json
import os


class UserProfile:
    """
    Persistent adaptive user model.
    Saves typing behavior across sessions.
    """

    def __init__(self, path="user_profile.json", alpha=0.05):
        self.path = path
        self.alpha = alpha

        # defaults
        self.avg_iki = None
        self.avg_pause = None
        self.avg_error = None

        self.var_iki = 1.0
        self.var_pause = 1.0
        self.var_error = 1.0

        # load if exists
        self.load()

    # =====================================================
    # UPDATE ONLINE
    # =====================================================
    def update(self, iki, pause, error):

        if self.avg_iki is None:
            self.avg_iki = iki
            self.avg_pause = pause
            self.avg_error = error
            return

        self.avg_iki = (1 - self.alpha) * self.avg_iki + self.alpha * iki
        self.avg_pause = (1 - self.alpha) * self.avg_pause + self.alpha * pause
        self.avg_error = (1 - self.alpha) * self.avg_error + self.alpha * error

        self.var_iki = (1 - self.alpha) * self.var_iki + self.alpha * (iki - self.avg_iki) ** 2
        self.var_pause = (1 - self.alpha) * self.var_pause + self.alpha * (pause - self.avg_pause) ** 2
        self.var_error = (1 - self.alpha) * self.var_error + self.alpha * (error - self.avg_error) ** 2

    # =====================================================
    # NORMALIZATION (for RL state)
    # =====================================================
    def normalize(self, iki, pause, error):

        return np.array([
            (iki - self.avg_iki) / (np.sqrt(self.var_iki) + 1e-6),
            (pause - self.avg_pause) / (np.sqrt(self.var_pause) + 1e-6),
            (error - self.avg_error) / (np.sqrt(self.var_error) + 1e-6),
        ])

    # =====================================================
    # SAVE PROFILE
    # =====================================================
    def save(self):

        data = {
            "avg_iki": self.avg_iki,
            "avg_pause": self.avg_pause,
            "avg_error": self.avg_error,
            "var_iki": self.var_iki,
            "var_pause": self.var_pause,
            "var_error": self.var_error,
            "alpha": self.alpha
        }

        with open(self.path, "w") as f:
            json.dump(data, f, indent=4)

    # =====================================================
    # LOAD PROFILE
    # =====================================================
    def load(self):

        if not os.path.exists(self.path):
            return

        try:
            with open(self.path, "r") as f:
                data = json.load(f)

            self.avg_iki = data["avg_iki"]
            self.avg_pause = data["avg_pause"]
            self.avg_error = data["avg_error"]

            self.var_iki = data.get("var_iki", 1.0)
            self.var_pause = data.get("var_pause", 1.0)
            self.var_error = data.get("var_error", 1.0)

            self.alpha = data.get("alpha", self.alpha)

        except Exception as e:
            print("⚠ Failed to load user profile:", e)

    # =====================================================
    # DEBUG
    # =====================================================
    def summary(self):
        return {
            "avg_iki": self.avg_iki,
            "avg_pause": self.avg_pause,
            "avg_error": self.avg_error
        }