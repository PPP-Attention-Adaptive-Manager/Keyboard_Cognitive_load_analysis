import pandas as pd
import numpy as np
from numba import njit
from joblib import Parallel, delayed
import multiprocessing

# ==========================================================
# 1. MEMORY-EFFICIENT LEVENSHTEIN (O(N) Space)
# ==========================================================
@njit(cache=True)
def levenshtein_fast(s1, s2):
    """Iterative Levenshtein with only two rows of memory."""
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    if len(s2) == 0:
        return len(s1)

    previous_row = np.arange(len(s2) + 1, dtype=np.int32)
    for i, c1 in enumerate(s1):
        current_row = np.zeros(len(s2) + 1, dtype=np.int32)
        current_row[0] = i + 1
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row[j + 1] = min(insertions, deletions, substitutions)
        previous_row = current_row

    return previous_row[-1]

# ==========================================================
# 2. OPTIMIZED PARALLEL WRAPPER
# ==========================================================
def compute_levenshtein_batch(pairs):
    """Processes a list of (str, str) tuples."""
    results = []
    for s1, s2 in pairs:
        results.append(levenshtein_fast(s1, s2))
    return results

def compute_sentence_error_optimized(df):
    # Only get the final state of USER_INPUT for each group to avoid redundant work
    # Grouping by ID and SENTENCE, then taking the last input
    grouped_df = df.groupby(["PARTICIPANT_ID", "TEST_SECTION_ID", "SENTENCE"], sort=False)["USER_INPUT"].last().reset_index()
    
    # --- CRITICAL OPTIMIZATION: Compute only UNIQUE string pairs ---
    # In many datasets, the same sentence or input appears multiple times.
    unique_pairs = grouped_df[["SENTENCE", "USER_INPUT"]].drop_duplicates().copy()
    
    # Prepare data for parallel processing (lists of strings are fast to serialize)
    tasks = list(zip(unique_pairs["SENTENCE"].values, unique_pairs["USER_INPUT"].values))
    
    n_cores = multiprocessing.cpu_count()
    # Chunk the tasks to reduce parallel overhead
    chunks = np.array_split(tasks, n_cores)
    
    print(f"🚀 Computing {len(tasks)} unique Levenshtein pairs on {n_cores} cores...")
    
    flat_results = Parallel(n_jobs=n_cores, backend="loky")(
        delayed(compute_levenshtein_batch)(chunk) for chunk in chunks
    )
    
    # Flatten results and map back to unique pairs
    unique_pairs["ERROR_RATE_ML"] = [item for sublist in flat_results for item in sublist]
    
    # Merge the distances back to the grouped_df
    result_df = grouped_df.merge(unique_pairs, on=["SENTENCE", "USER_INPUT"], how="left")
    
    return result_df[["PARTICIPANT_ID", "TEST_SECTION_ID", "SENTENCE", "ERROR_RATE_ML"]]

# ==========================================================
# 3. FAST FEATURE ENGINEERING
# ==========================================================
def build_features(df: pd.DataFrame, min_pause=1000) -> pd.DataFrame:
    df = df.copy()

    # Numeric safety (Vectorized)
    for col in ["PRESS_TIME", "RELEASE_TIME", "KEYCODE"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["PRESS_TIME", "RELEASE_TIME"])

    # Temporal features
    df["IKI"] = df.groupby(["PARTICIPANT_ID", "TEST_SECTION_ID"])["PRESS_TIME"].diff().fillna(0).clip(lower=0)
    df["HOLD_TIME"] = (df["RELEASE_TIME"] - df["PRESS_TIME"]).clip(lower=0)
    
    # Pre-calculate flags to avoid slow lambdas in .agg()
    df["IS_BACKSPACE"] = df["KEYCODE"].isin([8, 46]).astype(int)
    df["IS_LONG_PAUSE"] = (df["IKI"] > min_pause).astype(int)

    # String safety
    df["SENTENCE"] = df["SENTENCE"].fillna("").astype(str)
    df["USER_INPUT"] = df["USER_INPUT"].fillna("").astype(str)
    df["SENTENCE_LENGTH"] = df["SENTENCE"].str.len()

    # Compute ML error
    error_df = compute_sentence_error_optimized(df)

    # Merge distances back to main df
    df = df.merge(error_df, on=["PARTICIPANT_ID", "TEST_SECTION_ID", "SENTENCE"], how="left")

    # Aggregation (Removed slow lambdas)
    session_features = df.groupby(["PARTICIPANT_ID", "TEST_SECTION_ID"]).agg(
        MEAN_IKI=("IKI", "mean"),
        MEDIAN_IKI=("IKI", "median"),
        STD_IKI=("IKI", "std"),
        LONG_PAUSE_COUNT=("IS_LONG_PAUSE", "sum"),
        PAUSE_RATIO=("IS_LONG_PAUSE", "mean"),
        MEAN_HOLD_TIME=("HOLD_TIME", "mean"),
        MAX_HOLD_TIME=("HOLD_TIME", "max"),
        TOTAL_KEYSTROKES=("PRESS_TIME", "count"),
        BACKSPACE_COUNT=("IS_BACKSPACE", "sum"),
        MEAN_ERROR_RATE_ML=("ERROR_RATE_ML", "mean"),
        MEAN_SENTENCE_LENGTH=("SENTENCE_LENGTH", "mean")
    )

    # Post-aggregation math (Faster than lambda)
    session_features["IKI_CV"] = session_features["STD_IKI"] / (session_features["MEAN_IKI"] + 1e-9)
    session_features["BACKSPACE_RATIO"] = session_features["BACKSPACE_COUNT"] / (session_features["TOTAL_KEYSTROKES"] + 1e-9)
    session_features["KSPC_PROXY"] = session_features["TOTAL_KEYSTROKES"] / (session_features["MEAN_SENTENCE_LENGTH"] + 1e-9)

    return session_features.reset_index().fillna(0)