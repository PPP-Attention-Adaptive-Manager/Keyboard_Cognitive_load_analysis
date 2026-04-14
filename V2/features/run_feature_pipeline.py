import pandas as pd
from pathlib import Path

from preprocessing.temporal_builder import build_temporal_sequences
from features.feature_engineering import build_features
from proxies.proxy_definitions import build_all_proxies

# ==========================================================
# PATHS
# ==========================================================
INPUT_FILE = "data/raw/merged_first_sections.csv"
TEMP_FILE = "data/processed/temporal_sequences.csv"
FEATURES_FILE = "data/processed/session_features_nocog.csv"
OUTPUT_FILE = "data/processed/session_features.csv"

# ==========================================================
# PIPELINE
# ==========================================================

def main():
    features = None
    df = None

    # ------------------------------------------------------
    # CHECK STEP 2 CACHE (The Furthest Point)
    # ------------------------------------------------------
    if Path(FEATURES_FILE).exists():
        print("⏩ Step 2 cache found! Skipping Step 1 & 2...")
        print(f"📥 Loading: {FEATURES_FILE}")
        features = pd.read_csv(FEATURES_FILE)
    
    else:
        # ------------------------------------------------------
        # CHECK STEP 1 CACHE
        # ------------------------------------------------------
        if Path(TEMP_FILE).exists():
            print("⏩ Step 1 cache found! Skipping Step 1...")
            print(f"📥 Loading: {TEMP_FILE}")
            df = pd.read_csv(TEMP_FILE)
        else:
            # RUN STEP 1: RAW -> TEMPORAL
            print("📥 Loading raw data...")
            if not Path(INPUT_FILE).exists():
                raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")
            
            df = pd.read_csv(INPUT_FILE)
            df = df[pd.to_numeric(df["RELEASE_TIME"], errors="coerce").notna()]
            
            print(f"✔ Raw shape: {df.shape}")
            print("⏱ Building temporal sequences (Step 1)...")
            df = build_temporal_sequences(df)
            
            print(f"💾 Saving Step 1 cache → {TEMP_FILE}")
            df.to_csv(TEMP_FILE, index=False)

        # ------------------------------------------------------
        # RUN STEP 2: TEMPORAL -> FEATURES
        # ------------------------------------------------------
        print(f"✔ Temporal shape: {df.shape}")
        print("🧠 Building ML features (Step 2)...")
        features = build_features(df)
        
        print(f"💾 Saving Step 2 cache → {FEATURES_FILE}")
        features.to_csv(FEATURES_FILE, index=False)

    # ------------------------------------------------------
    # STEP 3: PROXIES (ALWAYS RUN)
    # ------------------------------------------------------
    print(f"✔ Features shape: {features.shape}")
    print("🧠 Building cognitive proxies (Step 3)...")
    
    # Always run proxy definitions on the features (cached or newly built)
    final_df = build_all_proxies(features)

    print(f"✔ Final dataset shape: {final_df.shape}")

    # ------------------------------------------------------
    # SAVE FINAL
    # ------------------------------------------------------
    print(f"💾 Saving final dataset → {OUTPUT_FILE}")
    final_df.to_csv(OUTPUT_FILE, index=False)

    print("✅ Pipeline complete")

if __name__ == "__main__":
    main()