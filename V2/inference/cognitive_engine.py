import pandas as pd
import numpy as np


# ==========================================================
# CONFIG
# ==========================================================

SMOOTH_ALPHA = 0.3


# ==========================================================
# COGNITIVE STATE ENGINE
# ==========================================================

class CognitiveEngine:
    """
    Converts behavioral features → latent cognitive state S_t
    """

    def __init__(self):
        self.user_states = {}

    # ======================================================
    # INIT STATE
    # ======================================================

    def _init_state(self):
        return {
            "cognitive_load": 0.5,
            "stress": 0.5,
            "attention": 0.5
        }

    # ======================================================
    # UPDATE RULES (CORE OF MODEL)
    # ======================================================

    def update_state(self, prev_state, x):
        """
        x = feature vector at time t
        """

        # ---------------------------
        # Cognitive Load dynamics
        # ---------------------------
        cognitive_load = (
            0.6 * x["STD_IKI"] +
            0.2 * x["PAUSE_RATIO"] +
            0.2 * x["IKI_CV"]
        )

        # ---------------------------
        # Stress dynamics
        # ---------------------------
        stress = (
            0.5 * x["BACKSPACE_RATIO"] +
            0.3 * x["MEAN_ERROR_RATE"] +
            0.2 * x["LONG_PAUSE_COUNT"]
        )

        # ---------------------------
        # Attention dynamics
        # ---------------------------
        attention = (
            1.0 - x["PAUSE_RATIO"]
        )

        # ======================================================
        # SMOOTH STATE TRANSITION (VERY IMPORTANT)
        # ======================================================

        new_state = {
            "cognitive_load": SMOOTH_ALPHA * cognitive_load +
                              (1 - SMOOTH_ALPHA) * prev_state["cognitive_load"],

            "stress": SMOOTH_ALPHA * stress +
                      (1 - SMOOTH_ALPHA) * prev_state["stress"],

            "attention": SMOOTH_ALPHA * attention +
                         (1 - SMOOTH_ALPHA) * prev_state["attention"],
        }

        return new_state

    # ======================================================
    # PROCESS ONE SEQUENCE
    # ======================================================

    def process_sequence(self, df_sequence: pd.DataFrame):
        """
        Converts full session sequence → cognitive trajectory
        """

        trajectory = []

        state = self._init_state()

        for _, row in df_sequence.iterrows():
            state = self.update_state(state, row)
            trajectory.append(state.copy())

        return pd.DataFrame(trajectory)

    # ======================================================
    # PROCESS ALL USERS
    # ======================================================

    def process_all(self, sequences: dict):
        """
        sequences = output of temporal_builder.py
        """

        results = {}

        for key, seq in sequences.items():
            results[key] = self.process_sequence(seq)

        return results