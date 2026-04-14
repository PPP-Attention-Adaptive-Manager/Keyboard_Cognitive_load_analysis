import pandas as pd
import numpy as np

def build_all_proxies(df):
    df = df.copy()
    
    # 1. PRE-CALCULATE ALL Z-SCORES & DIFFS (Vectorized)
    # We group once and transform multiple columns at once. 
    # This is orders of magnitude faster than a Python loop.
    print("⚡ Computing grouped Z-scores and differences...")
    
    group = df.groupby("PARTICIPANT_ID", sort=False)
    
    # Metrics needed for Z-scores
    z_cols = ["STD_IKI", "IKI_CV", "PAUSE_RATIO", "BACKSPACE_RATIO", "LONG_PAUSE_COUNT"]
    
    for col in z_cols:
        # Vectorized Z-Score: (x - mean) / std
        means = group[col].transform("mean")
        stds = group[col].transform("std")
        df[f"Z_{col}"] = (df[col] - means) / (stds + 1e-9)

    # Calculate Speed Change (Diff)
    df["SPEED_CHANGE"] = group["MEDIAN_IKI"].diff().abs().fillna(0)
    df["Z_SPEED_CHANGE"] = (df["SPEED_CHANGE"] - group["SPEED_CHANGE"].transform("mean")) / (group["SPEED_CHANGE"].transform("std") + 1e-9)

    # 2. COMPUTE RAW SCORES (Pure Vectorized Math)
    print("🧠 Computing Raw Proxy scores...")
    df["COGLOAD_RAW"] = (0.5 * df["Z_STD_IKI"] + 0.3 * df["Z_IKI_CV"] + 0.2 * df["Z_PAUSE_RATIO"])
    df["STRESS_RAW"] = (0.7 * df["Z_BACKSPACE_RATIO"] + 0.3 * df["Z_LONG_PAUSE_COUNT"])
    df["UNFOCUS_RAW"] = (0.6 * df["Z_SPEED_CHANGE"] + 0.4 * df["Z_PAUSE_RATIO"])

    # 3. SMOOTHING (Grouped Rolling Mean)
    # This replaces the smooth_series helper
    print("🌊 Applying smoothing...")
    raw_cols = ["COGLOAD_RAW", "STRESS_RAW", "UNFOCUS_RAW"]
    for col in raw_cols:
        # transform(lambda x...) with rolling is the fastest way to handle 160k+ groups
        df[col] = group[col].transform(lambda x: x.rolling(3, min_periods=1).mean())

    # 4. MIN-MAX SCALING (Grouped)
    print("⚖️ Applying final Min-Max scaling...")
    for col in raw_cols:
        g_min = group[col].transform("min")
        g_max = group[col].transform("max")
        proxy_name = col.replace("_RAW", "_PROXY")
        df[proxy_name] = (df[col] - g_min) / (g_max - g_min + 1e-9)

    return df