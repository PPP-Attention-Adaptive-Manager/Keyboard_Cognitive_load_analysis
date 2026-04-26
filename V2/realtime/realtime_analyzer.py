# ============================================================
# 🧠 REAL-TIME COGNITIVE LOAD ANALYZER
# ============================================================

import time
import threading
from collections import deque

import numpy as np
import pandas as pd
import joblib

from pynput import keyboard

import tkinter as tk
from tkinter import ttk


# ============================================================
# ⚙️ LOAD TRAINED MODELS
# ============================================================

LABEL_MODEL = joblib.load("../models/cogload_label_model.joblib")
RISK_MODEL = joblib.load("../models/cogload_risk_model.joblib")

FEATURES = [
    "IKT",
    "dwell_time",
    "typing_speed",
    "burstiness",
    "ikt_mean_5",
    "dwell_mean_5",
    "dwell_std_5",
    "is_alpha",
    "is_digit",
    "is_space",
    "is_punct",
    "is_backspace",
    "keycode_norm",
    "activity_density",
]

# ============================================================
# 🧠 REALTIME FEATURE BUFFER
# ============================================================

BUFFER_SIZE = 30

events = deque(maxlen=BUFFER_SIZE)
press_times = {}

last_event_time = None


# ============================================================
# 🧠 FEATURE ENGINEERING (LIVE VERSION)
# ============================================================

def build_features():

    if len(events) < 10:
        return None

    df = pd.DataFrame(events)

    df["ikt_mean_5"] = df["IKT"].rolling(5).mean()
    df["dwell_mean_5"] = df["dwell_time"].rolling(5).mean()
    df["dwell_std_5"] = df["dwell_time"].rolling(5).std()

    df["typing_speed"] = 1 / (df["IKT"] + 1e-6)
    df["burstiness"] = df["typing_speed"].rolling(5).std()

    df["activity_density"] = len(df) / max(df["IKT"].sum(), 1e-6)

    df = df.fillna(0)

    return df[FEATURES].iloc[-1:].values


# ============================================================
# 🧠 KEYBOARD LISTENER
# ============================================================

def on_press(key):

    global last_event_time

    t = time.time()

    try:
        k = key.char
    except:
        k = str(key)

    press_times[k] = t

    if last_event_time is None:
        ikt = 0
    else:
        ikt = t - last_event_time

    last_event_time = t

    events.append({
        "IKT": ikt,
        "dwell_time": 0,
        "is_alpha": int(str(k).isalpha()),
        "is_digit": int(str(k).isdigit()),
        "is_space": int(k == " "),
        "is_punct": int(not str(k).isalnum()),
        "is_backspace": int("backspace" in k.lower()),
        "keycode_norm": hash(k) % 1000 / 1000
    })


def on_release(key):

    t = time.time()

    try:
        k = key.char
    except:
        k = str(key)

    if k in press_times:
        dwell = t - press_times[k]

        if len(events):
            events[-1]["dwell_time"] = dwell


listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release
)

listener.start()

# ============================================================
# 🧠 STATE INTERPRETATION
# ============================================================

STATE_MAP = {
    0: "Focused",
    1: "Fatigued",
    2: "Overloaded"
}


# ============================================================
# 🧠 TKINTER GUI
# ============================================================

root = tk.Tk()
root.title("🧠 Cognitive Load Analyzer")
root.geometry("400x300")

title = ttk.Label(root, text="Real-Time Cognitive State", font=("Arial", 16))
title.pack(pady=10)

state_label = ttk.Label(root, text="Waiting...", font=("Arial", 20))
state_label.pack(pady=20)

risk_bar = ttk.Progressbar(root, length=300)
risk_bar.pack(pady=20)

risk_text = ttk.Label(root, text="Risk: 0.0")
risk_text.pack()

# ============================================================
# 🧠 REALTIME PREDICTION LOOP
# ============================================================

def update_prediction():

    feats = build_features()

    if feats is not None:

        label = LABEL_MODEL.predict(feats)[0]
        risk = RISK_MODEL.predict(feats)[0]

        state_label.config(text=STATE_MAP.get(label, "Unknown"))

        risk_bar["value"] = risk * 100
        risk_text.config(text=f"Risk: {risk:.2f}")

    root.after(500, update_prediction)


update_prediction()

# ============================================================
# 🚀 RUN APP
# ============================================================

root.mainloop()