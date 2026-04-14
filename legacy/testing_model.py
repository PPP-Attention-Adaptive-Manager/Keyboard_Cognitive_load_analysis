import tkinter as tk
import numpy as np
import time
import collections
import joblib

from dqn_agent import DQNAgent
from user_profile import UserProfile


# =====================================================
# SAFE NORMALIZATION
# =====================================================
def norm(x, k=1.0):
    return float(x / (x + k + 1e-6))


class CognitiveApp:

    def __init__(self, root):

        self.root = root
        self.root.title("Cognitive RL System (Research Mode)")
        self.root.geometry("800x600")

        # -------------------------
        # ML MODEL
        # -------------------------
        self.model = joblib.load("behavior_dynamics_model.pkl")
        self.scaler = joblib.load("behavior_scaler.pkl")

        # -------------------------
        # RL + PROFILE
        # -------------------------
        self.agent = DQNAgent(state_dim=3, action_dim=5)
        self.profile = UserProfile()

        # -------------------------
        # MEMORY
        # -------------------------
        self.press = []
        self.hold = []
        self.backspace = []

        self.history = collections.deque(maxlen=20)

        self.build_ui()

        self.text.bind("<KeyPress>", self.on_press)
        self.text.bind("<KeyRelease>", self.on_release)

        self.loop()

    # =====================================================
    def build_ui(self):

        self.text = tk.Text(self.root, height=10)
        self.text.pack(fill="both")

        self.out = tk.Text(self.root, height=25)
        self.out.pack(fill="both", expand=True)

    # =====================================================
    def log(self, text):
        self.out.insert("end", text + "\n")
        self.out.see("end")

    # =====================================================
    def on_press(self, e):
        self.press.append(time.time())

        if e.keysym in ["BackSpace", "Delete"]:
            self.backspace.append(time.time())

    def on_release(self, e):
        if self.press:
            self.hold.append(time.time() - self.press[-1])

    # =====================================================
    def features(self):

        now = time.time()
        window = 60

        p = [t for t in self.press if now - t < window]
        h = self.hold[-200:]

        if len(p) < 2:
            return np.zeros(9)

        iki = np.diff(p)

        median = np.median(iki)
        std = np.std(iki)
        cv = std / (np.mean(iki) + 1e-6)

        pause = np.sum(iki > 1.2) / (len(iki) + 1e-6)

        back = len(self.backspace)
        total = len(p)

        text_len = len(self.text.get("1.0", "end"))
        hold_mean = np.mean(h) if h else 0.1

        return np.array([
            median, std, cv,
            total, text_len,
            back, pause,
            hold_mean,
            window
        ])

    # =====================================================
    def ml_predict(self, f):
        x = np.array(f).reshape(1, -1)

        if x.shape[1] != self.scaler.n_features_in_:
            return 0.0

        return float(self.model.predict(self.scaler.transform(x))[0].mean())

    # =====================================================
    def rl_state(self, f):

        iki = f[0]
        pause = f[6]
        error = f[5] / (f[3] + 1e-6)

        self.profile.update(iki, pause, error)

        return self.profile.normalize(iki, pause, error)

    # =====================================================
    def loop(self):

        f = self.features()

        ml = self.ml_predict(f)
        state = self.rl_state(f)

        action = self.agent.act(state)

        # RL influence
        rl_score = (action - 2) / 2  # [-1, 1]
        rl_norm = norm(abs(rl_score))

        # ML normalization
        ml_norm = norm(ml)

        # combined cognitive load
        cog = norm(0.6 * ml_norm + 0.4 * rl_norm)

        # -------------------------
        # STRESS / FOCUS
        # -------------------------
        stress = norm(f[6])  # pauses
        focus = norm(1 - stress)

        # -------------------------
        # ADAPTATION METRIC
        # -------------------------
        similarity = np.exp(-np.linalg.norm(state - np.mean(self.profile.normalize(
            f[0], f[6], f[5]/(f[3]+1e-6)
        ))))

        adaptation = norm(similarity)

        # -------------------------
        # RL INTERNALS
        # -------------------------
        exploration = self.agent.epsilon

        # reward signal (stability)
        reward = 1 - abs(stress - focus)

        self.agent.remember(state, action, reward, state)
        self.agent.train()

        # -------------------------
        # OUTPUT (ALL NUMERICAL)
        # -------------------------
        self.log(
f"""
========================
COGNITIVE SYSTEM STATE
========================
Load        : {cog:.4f}
Stress      : {stress:.4f}
Focus       : {focus:.4f}
Error proxy : {norm(f[5]/(f[3]+1e-6)):.4f}

--- RL ---
RL score    : {rl_score:.4f}
RL norm     : {rl_norm:.4f}
Exploration : {exploration:.4f}

--- Adaptation ---
Adaptation  : {adaptation:.4f}

--- Raw ---
IKI median  : {norm(f[0]):.4f}
Pause ratio : {norm(f[6]):.4f}
Keystrokes  : {norm(f[3]):.4f}
========================
"""
        )

        self.root.after(500, self.loop)


if __name__ == "__main__":
    root = tk.Tk()
    app = CognitiveApp(root)
    root.mainloop()