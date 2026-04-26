import tkinter as tk
import time
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from V2.inference.realtime_features import RealtimeFeatureBuilder
from V2.personalization.user_profile import UserProfile
from V2.models.cognitive_model import CognitiveModel
from V2.utils.path import model_path


class CognitiveUI:

    def __init__(self):

        self.root = tk.Tk()
        self.root.title("Realtime Cognitive Load Monitor")
        self.root.geometry("900x700")

        # --------------------
        # MODEL STACK
        # --------------------
        self.builder = RealtimeFeatureBuilder()
        self.profile = UserProfile()
        self.model = CognitiveModel(
            self.profile,
            model_path("behavior_model.pth")
        )

        # --------------------
        # TEXT INPUT
        # --------------------
        self.text = tk.Text(self.root, height=10, font=("Consolas", 14))
        self.text.pack(fill="x", padx=10, pady=10)

        self.text.bind("<KeyPress>", self.on_press)
        self.text.bind("<KeyRelease>", self.on_release)

        # --------------------
        # METRICS DISPLAY
        # --------------------
        self.metrics = {}
        for name in ["cognitive_load", "stress", "unfocus"]:
            lbl = tk.Label(self.root, text=f"{name}: 0.000", font=("Arial", 14))
            lbl.pack(anchor="w", padx=20)
            self.metrics[name] = lbl

        # --------------------
        # SMOOTHING (EMA)
        # --------------------
        self.alpha = 0.3
        self.smoothed = {
            "cognitive_load": 0.0,
            "stress": 0.0,
            "unfocus": 0.0,
        }

        # --------------------
        # HISTORY (FOR PLOTS)
        # --------------------
        self.history_size = 100
        self.history = {
            "cognitive_load": deque(maxlen=self.history_size),
            "stress": deque(maxlen=self.history_size),
            "unfocus": deque(maxlen=self.history_size),
        }

        # --------------------
        # PLOT SETUP
        # --------------------
        self.fig, self.ax = plt.subplots(figsize=(8, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # --------------------
        # FEATURE DISPLAY
        # --------------------
        self.feature_label = tk.Label(
            self.root,
            text="Waiting for typing...",
            justify="left",
            font=("Consolas", 10),
        )
        self.feature_label.pack(pady=10)

    # -----------------------------------
    # KEY EVENTS
    # -----------------------------------
    def on_press(self, event):
        self.builder.key_press(event.keysym.lower(), time.time())
        self.update_model()

    def on_release(self, event):
        self.builder.key_release(event.keysym.lower(), time.time())

    # -----------------------------------
    # UPDATE MODEL
    # -----------------------------------
    def update_model(self):

        features = self.builder.build_features()
        if features is None:
            return

        self.profile.update(features)
        result = self.model.predict(features)

        # --------------------
        # EMA smoothing + history
        # --------------------
        for k in self.history.keys():

            raw = result[k]

            self.smoothed[k] = (
                self.alpha * raw +
                (1 - self.alpha) * self.smoothed[k]
            )

            self.history[k].append(self.smoothed[k])

        # --------------------
        # update UI labels
        # --------------------
        for k in self.history.keys():
            self.metrics[k].config(text=f"{k}: {self.smoothed[k]:.3f}")

        # --------------------
        # update feature view
        # --------------------
        text = ""
        for k, v in features.iloc[0].items():
            text += f"{k}: {v:.4f}\n"
        self.feature_label.config(text=text)

        # --------------------
        # update graph
        # --------------------
        self.update_plot()

        # --------------------
        # ALERT SYSTEM
        # --------------------
        if self.smoothed["stress"] > 0.85 and self.smoothed["cognitive_load"] > 0.7:
            self.root.configure(bg="#ffdddd")
        else:
            self.root.configure(bg="white")

    # -----------------------------------
    # PLOT UPDATE
    # -----------------------------------
    def update_plot(self):

        self.ax.clear()

        self.ax.plot(list(self.history["cognitive_load"]), label="cognitive_load")
        self.ax.plot(list(self.history["stress"]), label="stress")
        self.ax.plot(list(self.history["unfocus"]), label="unfocus")

        self.ax.set_ylim(0, 1)
        self.ax.set_title("Real-Time Cognitive State")
        self.ax.legend()

        self.canvas.draw()

    # -----------------------------------
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = CognitiveUI()
    app.run()