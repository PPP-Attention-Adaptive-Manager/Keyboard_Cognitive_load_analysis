# rl/behavior_rl_env.py

import numpy as np


class BehaviorRLEnv:
    """
    RL layer that learns correction weights over:
    - ML predictions
    - static proxies
    - user profile memory
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.state = None
        return self.state

    # ----------------------------------------------------------
    # STATE BUILDER
    # ----------------------------------------------------------

    def build_state(self, ml_features, proxies, user_vector, time_features):
        """
        Combines all signals into RL state
        """

        self.state = np.concatenate([
            ml_features,      # ML behavioral features
            proxies,          # cognitive proxies
            user_vector,      # long-term memory
            time_features     # session index, recency, etc.
        ])

        return self.state

    # ----------------------------------------------------------
    # ACTION SPACE
    # ----------------------------------------------------------

    def step(self, action, ml_output, proxy_output, user_vector):
        """
        action = weighting vector learned by RL

        action = [w_ml, w_proxy, w_user]
        """

        w_ml, w_proxy, w_user = action

        # normalize weights
        s = w_ml + w_proxy + w_user + 1e-9
        w_ml /= s
        w_proxy /= s
        w_user /= s

        # final prediction
        final_output = (
            w_ml * ml_output +
            w_proxy * proxy_output +
            w_user * user_vector.mean()
        )

        # ------------------------------------------------------
        # reward (example heuristic)
        # ------------------------------------------------------

        reward = -np.var(final_output)  # stability reward

        return final_output, reward