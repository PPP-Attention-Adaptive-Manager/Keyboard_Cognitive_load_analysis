import pandas as pd
import numpy as np

def build_temporal_sequences(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ==========================================================
    # 🔒 HARD TYPE SAFETY (CRITICAL FIX)
    # ==========================================================
    print("step 0: 🔒 Ensuring type safety for temporal calculations...")
    for col in ["PRESS_TIME", "RELEASE_TIME"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["PRESS_TIME", "RELEASE_TIME"])

    df["PRESS_TIME"] = df["PRESS_TIME"].astype(float)
    df["RELEASE_TIME"] = df["RELEASE_TIME"].astype(float)

    # ==========================================================
    # SORT (IMPORTANT FOR TEMPORAL LOGIC)
    # ==========================================================
    print("step 1: 🔀 Sorting data for temporal logic...")
    df = df.sort_values(
        ["PARTICIPANT_ID", "TEST_SECTION_ID", "PRESS_TIME"]
    )

    # ==========================================================
    # IKI (INTER KEY INTERVAL)
    # ==========================================================
    print("step 2: ⏱ Computing Inter-Key Interval (IKI)...")
    df["IKI"] = (
        df.groupby(["PARTICIPANT_ID", "TEST_SECTION_ID"])["PRESS_TIME"]
        .diff()
        .fillna(0)
        .clip(lower=0)
    )

    # ==========================================================
    # HOLD TIME (NOW SAFE)
    # ==========================================================
    print("step 3: ⌛ Computing Hold Time...")
    df["HOLD_TIME"] = (
        df["RELEASE_TIME"] - df["PRESS_TIME"]
    ).clip(lower=0)

    return df