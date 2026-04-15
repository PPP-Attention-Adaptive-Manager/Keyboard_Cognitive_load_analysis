import numpy as np
from collections import defaultdict


class UserProfile:

    def __init__(self):

        self.history = defaultdict(list)

    # -------------------------
    # Update baseline
    # -------------------------
    def update(self, feature_df):

        row = feature_df.iloc[0].to_dict()

        for k, v in row.items():
            self.history[k].append(v)

    # -------------------------
    # Stats
    # -------------------------
    def get_mean(self, key):

        values = self.history[key]

        if len(values) < 5:
            return 0

        return np.mean(values)

    def get_std(self, key):

        values = self.history[key]

        if len(values) < 5:
            return 1

        return np.std(values) + 1e-6