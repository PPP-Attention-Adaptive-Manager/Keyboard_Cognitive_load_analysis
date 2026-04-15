import tkinter as tk
import time

from V2.inference.realtime_features import RealtimeFeatureBuilder
from V2.personalization.user_profile import UserProfile
from V2.models.cognitive_model import CognitiveModel


class CognitiveUI:

    def __init__(self):

        self.root = tk.Tk()
        self.root.title("Realtime Cognitive Load Monitor")
        self.root.geometry("700x500")

        # --------------------
        # MODEL STACK
        # --------------------
        self.builder = RealtimeFeatureBuilder()
        self.profile = UserProfile()
        self.model = CognitiveModel(self.profile)

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

        for name in ["cognitive_load", "focus", "fatigue", "stress"]:
            lbl = tk.Label(self.root,
                           text=f"{name}: 0",
                           font=("Arial", 14))
            lbl.pack(anchor="w", padx=20)
            self.metrics[name] = lbl

        # Feature display
        self.feature_label = tk.Label(
            self.root,
            text="Waiting for typing...",
            justify="left",
            font=("Consolas", 11),
        )
        self.feature_label.pack(pady=10)

    # -----------------------------------
    # KEY EVENTS
    # -----------------------------------

    def on_press(self, event):

        key = event.keysym.lower()
        t = time.time()

        self.builder.key_press(key, t)
        self.update_model()

    def on_release(self, event):

        key = event.keysym.lower()
        t = time.time()

        self.builder.key_release(key, t)

    # -----------------------------------
    # MODEL UPDATE
    # -----------------------------------

    def update_model(self):

        features = self.builder.build_features()

        if features is None:
            return

        # update baseline
        self.profile.update(features)

        result = self.model.predict(features)

        # update labels
        for k, v in result.items():
            self.metrics[k].config(text=f"{k}: {v:.3f}")

        # show raw features
        text = ""
        for k, v in features.iloc[0].items():
            text += f"{k}: {v:.4f}\n"

        self.feature_label.config(text=text)

    # -----------------------------------

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = CognitiveUI()
    app.run()