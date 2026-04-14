import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob
from tqdm import tqdm

# ==========================================================
# CONFIG
# ==========================================================

DATA_PATTERN = "../../../../Dataset/keystrokes/files/*.txt"
OUTPUT_FILE = Path("data/processed/sessions_raw.csv")

N_SECTIONS = 2
PAUSE_THRESHOLD = 3000
MAX_WORKERS = 12


# ==========================================================
# 1. LOAD SINGLE FILE
# ==========================================================

def load_file(file_path):
    try:
        df = pd.read_csv(
            file_path,
            sep="\t",
            encoding="latin1",
            on_bad_lines="skip"
        )
    except:
        return None

    required = {"PARTICIPANT_ID", "TEST_SECTION_ID", "PRESS_TIME", "RELEASE_TIME"}
    if not required.issubset(df.columns):
        return None

    # keep only first sections
    first_sections = (
        df["TEST_SECTION_ID"]
        .drop_duplicates()
        .head(N_SECTIONS)
        .tolist()
    )

    df = df[df["TEST_SECTION_ID"].isin(first_sections)]

    return df


# ==========================================================
# 2. PARALLEL MERGE
# ==========================================================

def merge_dataset():
    files = glob.glob(DATA_PATTERN)
    collected = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(load_file, f) for f in files]

        for f in tqdm(as_completed(futures), total=len(futures)):
            df = f.result()
            if df is not None:
                collected.append(df)

    return pd.concat(collected, ignore_index=True)


# ==========================================================
# 3. SESSION CONSTRUCTION (EVENT-LEVEL)
# ==========================================================

def build_sessions(df):
    df = df.copy()

    # numeric safety
    df["PRESS_TIME"] = pd.to_numeric(df["PRESS_TIME"], errors="coerce")
    df["RELEASE_TIME"] = pd.to_numeric(df["RELEASE_TIME"], errors="coerce")

    df = df.dropna(subset=["PRESS_TIME", "RELEASE_TIME"])

    # sort
    df = df.sort_values(
        ["PARTICIPANT_ID", "TEST_SECTION_ID", "PRESS_TIME"]
    )

    # inter-key interval
    df["IKI"] = (
        df.groupby(["PARTICIPANT_ID", "TEST_SECTION_ID"])["PRESS_TIME"]
        .diff()
        .fillna(0)
        .clip(lower=0)
    )

    # session segmentation
    df["NEW_SESSION"] = (df["IKI"] > PAUSE_THRESHOLD).astype(int)

    df["SESSION_ID"] = (
        df.groupby(["PARTICIPANT_ID", "TEST_SECTION_ID"])["NEW_SESSION"]
        .cumsum()
    )

    return df


# ==========================================================
# 4. MAIN
# ==========================================================

def main():
    print("📥 Loading raw data...")
    df = merge_dataset()

    print("🧠 Building sessions...")
    df = build_sessions(df)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ Saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()