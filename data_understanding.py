import pandas as pd

# Load a chunk because the file is 1GB+
df_chunk = pd.read_csv("session_cogload_metrics.csv", nrows=1000000)
print("backspace count=", df_chunk["BACKSPACE_COUNT"].max())
