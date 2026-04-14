import glob
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# -----------------------------
# Configuration
# -----------------------------
N_SECTIONS = 2
ROWS_PER_SECTION = None
MAX_WORKERS = 12

BASE_DIR = Path(__file__).resolve().parent
TXT_PATH = "../../Dataset/keystrokes/files/*.txt"
OUTPUT_FILE = BASE_DIR / "merged_first_sections.csv"


# -----------------------------
# Worker function
# -----------------------------
def process_file(file):
    try:
        df = pd.read_csv(
            file,
            sep="\t",
            quoting=3,
            on_bad_lines='skip',
            encoding='latin1'
        )
    except Exception as e:
        return None, f"Error reading {file}: {e}"

    if 'TEST_SECTION_ID' not in df.columns:
        return None, f"Skipping {file}: missing TEST_SECTION_ID"

    first_sections = df['TEST_SECTION_ID'].drop_duplicates().head(N_SECTIONS).tolist()
    df_filtered = df[df['TEST_SECTION_ID'].isin(first_sections)]

    if ROWS_PER_SECTION is not None:
        df_filtered = df_filtered.groupby('TEST_SECTION_ID').head(ROWS_PER_SECTION)

    return df_filtered, f"{file}: sections {first_sections}, rows {len(df_filtered)}"


# -----------------------------
# MAIN (IMPORTANT FOR WINDOWS)
# -----------------------------
def main():
    txt_files = glob.glob(str(TXT_PATH))

    if not txt_files:
        raise FileNotFoundError(f"No files found matching {TXT_PATH}")

    filtered_dfs = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_file, file) for file in txt_files]

        for future in tqdm(as_completed(futures),
                           total=len(futures),
                           desc="Processing files"):
            df_result, msg = future.result()

            if msg:
                tqdm.write(msg)

            if df_result is not None:
                filtered_dfs.append(df_result)

    if filtered_dfs:
        merged_df = pd.concat(filtered_dfs, ignore_index=True)
        merged_df.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ Merged {len(filtered_dfs)} files into {OUTPUT_FILE}")
    else:
        print("⚠️ No data to merge.")


# REQUIRED ON WINDOWS
if __name__ == "__main__":
    main()