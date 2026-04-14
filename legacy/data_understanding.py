import pandas as pd
df = pd.read_csv("merged_first_sections.csv", low_memory=False,nrows=1000000)

print(df.columns[1], df.columns[6])
# Load a chunk because the file is 1GB+
# df_chunk = pd.read_csv("session_cogload_metrics.csv", nrows=1000000)
# print("backspace count=", df_chunk["BACKSPACE_COUNT"].max())
bad_release = df[pd.to_numeric(df["RELEASE_TIME"], errors="coerce").isna()]

print(bad_release[["RELEASE_TIME"]].head(20))
print("Bad rows:", len(bad_release))
print("Columns:", df.columns.tolist())
# ==========================================================
# CREATE MISSING BACKSPACE FLAG
# ==========================================================
BACKSPACE_CODES = [8, 46]

df["KEYCODE"] = pd.to_numeric(df["KEYCODE"], errors="coerce")

df["IS_BACKSPACE"] = df["KEYCODE"].isin(BACKSPACE_CODES).astype(int)
print("Backspace count:", df["IS_BACKSPACE"].sum())