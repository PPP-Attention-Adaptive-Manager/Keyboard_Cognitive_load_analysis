import numpy as np
import pandas as pd
from collections import deque


class RealtimeFeatureBuilder:
    """
    Real-time cognitive feature extractor for typing behavior.
    Produces stable, adaptive, user-independent signals.
    """

    def __init__(self, window_size=80):
        self.window_size = window_size

        # temporal buffers
        self.ikis = deque(maxlen=window_size)
        self.holds = deque(maxlen=window_size)
        self.sentence_lengths = deque(maxlen=20)

        # counters
        self.backspaces = 0
        self.errors = 0
        self.total_keys = 0
        self.sentences = 0
        self.pause_count = 0

        # sentence tracking
        self.current_sentence_len = 0

        # timing
        self.last_press = None
        self.press_times = {}

    # ==========================================================
    # KEY EVENTS
    # ==========================================================

    def key_press(self, key, t):

        self.press_times[key] = t

        # ===============================
        # IKI (inter-key interval)
        # ===============================
        if self.last_press is not None:
            iki = t - self.last_press
            self.ikis.append(iki)

            # adaptive pause detection
            if len(self.ikis) > 10:
                baseline = np.mean(self.ikis)
                if iki > 2.0 * baseline:
                    self.pause_count += 1

        self.last_press = t

        # ===============================
        # counters
        # ===============================
        self.total_keys += 1
        self.current_sentence_len += 1

        if key == "backspace":
            self.backspaces += 1
            self.errors += 1

        # ===============================
        # sentence boundary detection
        # ===============================
        if key in ["enter", ".", "!", "?"]:
            self.sentences += 1

            if self.current_sentence_len > 0:
                self.sentence_lengths.append(self.current_sentence_len)

            self.current_sentence_len = 0

    def key_release(self, key, t):

        if key in self.press_times:
            hold = t - self.press_times[key]
            self.holds.append(hold)

    # ==========================================================
    # FEATURE EXTRACTION
    # ==========================================================

    def build_features(self):

        if len(self.ikis) < 10:
            return None

        ikis = np.array(self.ikis)
        holds = np.array(self.holds) if len(self.holds) > 0 else np.array([0])

        mean_iki = np.mean(ikis)
        std_iki = np.std(ikis)

        # ======================================================
        # sentence length (FIXED: no cumulative bias)
        # ======================================================
        if len(self.sentence_lengths) > 0:
            mean_sentence_length = np.mean(self.sentence_lengths)
        else:
            mean_sentence_length = self.current_sentence_len

        # ======================================================
        # real-time cognitive features
        # ======================================================
        features = {
            # motor control
            "MEAN_HOLD_TIME": np.mean(holds),
            "IKI_CV": std_iki / (mean_iki + 1e-9),

            # correction behavior
            "BACKSPACE_RATIO": self.backspaces / (self.total_keys + 1e-9),
            "MEAN_ERROR_RATE_ML": self.errors / (self.total_keys + 1e-9),

            # structural behavior
            "MEAN_SENTENCE_LENGTH": mean_sentence_length,

            # temporal instability
            "PAUSE_RATIO": self.pause_count / (len(self.ikis) + 1e-9),
        }

        return pd.DataFrame([features])