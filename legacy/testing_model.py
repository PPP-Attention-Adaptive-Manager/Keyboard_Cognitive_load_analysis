import tkinter as tk
from tkinter import ttk
import joblib
import numpy as np
import pandas as pd
import collections
import time

# -----------------------------
# Configuration & Loading
# -----------------------------
def load_model_and_scaler():
    try:
        model = joblib.load('rf_cogload_model_complex.pkl')
        scaler = joblib.load('feature_scaler_complex.pkl')
        # Exact feature names from your training session
        f_names = [
            'MEDIAN_IKI', 'STD_IKI', 'IKI_CV', 'TOTAL_KEYSTROKES',
            'ELAPSED_TIME', 'MEAN_SENTENCE_LENGTH', 'BACKSPACE_RATIO',
            'MEAN_HOLD_TIME', 'PAUSE_RATIO', 'PAUSE_PER_MIN'
        ]
        return model, scaler, f_names
    except FileNotFoundError:
        print("Error: Model files not found. Please ensure 'rf_cogload_model_complex.pkl' and 'feature_scaler_complex.pkl' are in this folder.")
        exit()

# -----------------------------
# GUI Application
# -----------------------------
class CognitiveLoadApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hybrid Cognitive Load Monitor (ML + Proxy)")
        self.root.geometry("900x700")
        self.root.configure(bg="#f8f9fa")
        
        # Data Tracking
        self.press_data = []  
        self.hold_data = []   
        self.key_press_map = {}
        self.backspace_log = [] 
        self.start_time = time.time()
        
        # Smoothing window for the Gauge
        self.recent_preds = collections.deque(maxlen=15)
        
        # Load Resources
        self.model, self.scaler, self.feature_names = load_model_and_scaler()

        self.setup_ui()
        
        # Event Bindings
        self.text_area.bind("<KeyPress>", self.on_key_press)
        self.text_area.bind("<KeyRelease>", self.on_key_release)

    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        container = ttk.Frame(self.root, padding="30")
        container.pack(fill="both", expand=True)

        # Header
        ttk.Label(container, text="Real-Time Cognitive Workload", font=("Segoe UI", 22, "bold")).pack(pady=(0, 5))
        ttk.Label(container, text="Logic: 60s Sliding Window | Method: Hybrid (RF + Proxy)", 
                  font=("Segoe UI", 10), foreground="#6c757d").pack(pady=(0, 20))
        
        # Text Input Area
        self.text_area = tk.Text(container, height=12, font=("Consolas", 13), 
                                 undo=True, padx=15, pady=15, relief="flat", highlightthickness=1)
        self.text_area.pack(fill="both", expand=True, pady=10)
        self.text_area.focus_set()

        # Gauge / Progress Bar
        self.load_var = tk.DoubleVar(value=0)
        self.progress = ttk.Progressbar(container, variable=self.load_var, maximum=10)
        self.progress.pack(fill="x", pady=15)

        # Prediction Labels
        self.prediction_label = ttk.Label(container, text="Calculated Load: 0.00", font=("Segoe UI", 20, "bold"))
        self.prediction_label.pack(pady=5)
        
        # Detailed Stats Footer
        self.stats_frame = ttk.Frame(container)
        self.stats_frame.pack(fill="x", pady=10)
        
        self.ml_label = ttk.Label(self.stats_frame, text="ML: 0.00", font=("Consolas", 10))
        self.ml_label.pack(side="left", padx=10)
        
        self.proxy_label = ttk.Label(self.stats_frame, text="Proxy: 0.00", font=("Consolas", 10))
        self.proxy_label.pack(side="left", padx=10)

        self.metrics_label = ttk.Label(self.stats_frame, text="Keys/min: 0 | Pauses: 0", font=("Consolas", 10))
        self.metrics_label.pack(side="right", padx=10)

        self.update_prediction()

    def on_key_press(self, event):
        now = time.time()
        self.press_data.append(now)
        self.key_press_map[event.keysym] = now
        # Detects Backspace or Delete as errors
        if event.keysym in ['BackSpace', 'Delete']:
            self.backspace_log.append(now)

    def on_key_release(self, event):
        now = time.time()
        if event.keysym in self.key_press_map:
            duration = now - self.key_press_map[event.keysym]
            self.hold_data.append((now, duration))
            del self.key_press_map[event.keysym]

    def calculate_manual_proxy(self, f):
        """
        Exact mirror of the training script formula:
        f[5]: MEAN_SENTENCE_LENGTH, f[6]: BACKSPACE_RATIO, f[7]: MEAN_HOLD_TIME, 
        f[8]: PAUSE_RATIO, f[9]: PAUSE_PER_MIN
        """
        try:
            # 1. log1p(MEAN_HOLD_TIME * PAUSE_RATIO)
            term1 = 0.3 * np.log1p(f[7] * f[8])
            
            # 2. sqrt(MAX_HOLD_TIME) 
            # (In real-time, we use the max from the current hold_data buffer)
            recent_holds = [d for t, d in self.hold_data if time.time() - t < 60]
            max_hold = max(recent_holds) if recent_holds else f[7]
            term2 = 0.2 * np.sqrt(max_hold)
            
            # 3. PAUSE_PER_MIN ^ 1.5
            term3 = 0.2 * (f[9] ** 1.5)
            
            # 4. tanh(LONG_PAUSE_COUNT)
            # Note: f[9] is PAUSE_PER_MIN. In a 1min window, Count = Rate.
            term4 = 0.1 * np.tanh(f[9])
            
            # 5. log1p(MEAN_SENTENCE_LENGTH * (1 + BACKSPACE_RATIO))
            term5 = 0.2 * np.log1p(f[5] * (1 + f[6]))
            
            return term1 + term2 + term3 + term4 + term5
        except Exception as e:
            print(f"Proxy Calc Error: {e}")
            return 0

    def extract_sliding_features(self):
        now = time.time()
        window = 60  # The "Logical Window"
        
        # 1. Filter Window
        recent_presses = [t for t in self.press_data if now - t < window]
        recent_holds = [d for t, d in self.hold_data if now - t < window]
        recent_backspaces = [t for t in self.backspace_log if now - t < window]
        
        # 2. Prune old data to keep memory clean
        self.press_data = [t for t in self.press_data if now - t < window * 2]
        self.hold_data = [(t, d) for t, d in self.hold_data if now - t < window * 2]
        self.backspace_log = [t for t in self.backspace_log if now - t < window * 2]

        # 3. Timing/IKI Logic
        if len(recent_presses) > 2:
            ikis = np.diff(recent_presses)
            median_iki = np.median(ikis)
            std_iki = np.std(ikis)
            iki_cv = std_iki / np.mean(ikis) if np.mean(ikis) > 0 else 0
            
            # Pause detection (> 1.5s is a long cognitive stall)
            pauses = [i for i in ikis if i > 1.5]
            pause_count = len(pauses)
            pause_ratio = sum(pauses) / window
        else:
            median_iki, std_iki, iki_cv = 0.5, 0.1, 0.2
            pause_count, pause_ratio = 0, 0

        # 4. Rates
        total_keystrokes = len(recent_presses)
        backspace_ratio = len(recent_backspaces) / total_keystrokes if total_keystrokes > 0 else 0
        mean_hold_time = np.mean(recent_holds) if recent_holds else 0.1
        
        content = self.text_area.get("1.0", tk.END).strip()
        words = content.split()[-20:] # Look at the tail end of text
        mean_sentence_length = np.mean([len(w) for w in words]) if words else 0
        
        pause_per_min = pause_count * (60 / window)

        return [
            median_iki, std_iki, iki_cv, float(total_keystrokes),
            float(window), mean_sentence_length, backspace_ratio,
            mean_hold_time, pause_ratio, pause_per_min
        ]

    def update_prediction(self):
        # A. Feature Extraction
        features = self.extract_sliding_features()
        
        # B. ML Prediction
        X_df = pd.DataFrame([features], columns=self.feature_names)
        scaled_features = self.scaler.transform(X_df)
        ml_val = self.model.predict(scaled_features)[0]
        
        # C. Manual Proxy Prediction
        proxy_val = self.calculate_manual_proxy(features)
        
        # D. Hybrid Average & Normalization
        # Average the ML insight (learned from 16M rows) with the Proxy (Hard logic)
        hybrid_raw = (ml_val + proxy_val) / 2
        
        # Normalize to 0-10 scale (Assuming proxy/ML range 1-7)
        normalized = np.clip((hybrid_raw / 6.0) * 10, 0, 10)
        
        # E. Smoothing for the GUI
        self.recent_preds.append(normalized)
        smoothed_val = np.mean(self.recent_preds)

        # F. Handle Idle State
        now = time.time()
        is_idle = False
        if self.press_data and (now - self.press_data[-1] > 60):
            smoothed_val = 0
            is_idle = True

        # G. Update UI
        self.load_var.set(smoothed_val)
        self.prediction_label.config(text=f"Hybrid Load: {smoothed_val:.2f}")
        
        # Color Feedback
        if is_idle: color = "#6c757d"
        elif smoothed_val < 3.5: color = "#28a745" # Green
        elif smoothed_val < 7: color = "#fd7e14"   # Orange
        else: color = "#dc3545"                     # Red
        
        self.prediction_label.config(foreground=color)
        
        # Stats Update
        self.ml_label.config(text=f"ML Predict: {ml_val:.2f}")
        self.proxy_label.config(text=f"Proxy Math: {proxy_val:.2f}")
        self.metrics_label.config(text=f"Keys: {features[3]} | Pauses/Min: {features[9]:.1f} | Err: {features[6]:.1%}")

        # Re-run in 1 second
        self.root.after(1000, self.update_prediction)

if __name__ == "__main__":
    root = tk.Tk()
    app = CognitiveLoadApp(root)
    root.mainloop()