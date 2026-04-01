import glob
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# -----------------------------
# Configuration
# -----------------------------
N_SECTIONS = 2
ROWS_PER_SECTION = None
TXT_PATH = "/pathtokeystrokes/Keystrokes/files/*.txt"
OUTPUT_FILE = "merged_first_sections.csv"
MAX_WORKERS = 12   # adjust based on your CPU (e.g. n_cores - 1)

# -----------------------------
# Worker function (runs in parallel)
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

    msg = f"{file}: sections {first_sections}, rows {len(df_filtered)}"
    return df_filtered, msg

# -----------------------------
# Step 1: Find files
# -----------------------------
txt_files = glob.glob(TXT_PATH)
if not txt_files:
    raise FileNotFoundError(f"No files found matching {TXT_PATH}")

# -----------------------------
# Step 2: Parallel processing
# -----------------------------
filtered_dfs = []

with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(process_file, file) for file in txt_files]

    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
        df_result, msg = future.result()

        if msg:
            tqdm.write(msg)

        if df_result is not None:
            filtered_dfs.append(df_result)

# -----------------------------
# Step 3: Merge results
# -----------------------------
if filtered_dfs:
    merged_df = pd.concat(filtered_dfs, ignore_index=True)
    merged_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Merged {len(filtered_dfs)} files into {OUTPUT_FILE}")
else:
    print("No data to merge.")