# rl/simple_policy.py

import numpy as np


class SimpleAdaptivePolicy:
    """
    Learns weighting strategy over time (A/B/C fusion)
    """

    def __init__(self):
        self.weights = np.array([0.5, 0.3, 0.2])
        self.lr = 0.05

    def act(self, ml, proxy, user):
        return self.weights

    def update(self, reward_signal):
        """
        Simple gradient-free adaptation
        """

        if reward_signal > 0:
            self.weights += self.lr * np.random.randn(3)
        else:
            self.weights -= self.lr * np.random.randn(3)

        self.weights = np.clip(self.weights, 0.05, 1.0)
        self.weights /= self.weights.sum()