import pandas as pd
import numpy as np
from numba import njit
from joblib import Parallel, delayed
import multiprocessing


# ==========================================================
# ENTROPY
# ==========================================================
def entropy(series):
    p = series.value_counts(normalize=True)
    return -(p * np.log2(p + 1e-9)).sum()


# ==========================================================
# LEVENSHTEIN (FAST)
# ==========================================================
@njit(cache=True)
def levenshtein_fast(s1, s2):
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    if len(s2) == 0:
        return len(s1)

    prev = np.arange(len(s2) + 1, dtype=np.int32)

    for i, c1 in enumerate(s1):
        curr = np.zeros(len(s2) + 1, dtype=np.int32)
        curr[0] = i + 1

        for j, c2 in enumerate(s2):
            ins = prev[j + 1] + 1
            dele = curr[j] + 1
            sub = prev[j] + (c1 != c2)
            curr[j + 1] = min(ins, dele, sub)

        prev = curr

    return prev[-1]


# ==========================================================
# ERROR COMPUTATION
# ==========================================================
def compute_error(df):

    grouped = df.groupby(
        ["PARTICIPANT_ID", "TEST_SECTION_ID", "SENTENCE"]
    )["USER_INPUT"].last().reset_index()

    unique = grouped[["SENTENCE", "USER_INPUT"]].drop_duplicates()

    tasks = list(zip(unique["SENTENCE"], unique["USER_INPUT"]))

    n_cores = multiprocessing.cpu_count()
    chunks = np.array_split(tasks, n_cores)

    def worker(chunk):
        return [levenshtein_fast(a, b) for a, b in chunk]

    results = Parallel(n_jobs=n_cores, backend="loky")(
        delayed(worker)(chunk) for chunk in chunks
    )

    flat = [x for sub in results for x in sub]
    unique["ERROR_RATE_ML"] = flat

    return grouped.merge(unique, on=["SENTENCE", "USER_INPUT"], how="left")


# ==========================================================
# MAIN FEATURE ENGINEERING
# ==========================================================
def build_features(df: pd.DataFrame):

    df = df.copy()

    # ==========================================================
    # CLEAN
    # ==========================================================
    for col in ["PRESS_TIME", "RELEASE_TIME", "KEYCODE"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["PRESS_TIME", "RELEASE_TIME"])

    df["SENTENCE"] = df["SENTENCE"].fillna("").astype(str)
    df["USER_INPUT"] = df["USER_INPUT"].fillna("").astype(str)

    # ==========================================================
    # ERROR (INSIDE FEATURE ENGINEERING)
    # ==========================================================
    err = compute_error(df)

    df = df.merge(
        err,
        on=["PARTICIPANT_ID", "TEST_SECTION_ID", "SENTENCE"],
        how="left"
    )

    df["ERROR_RATE_ML"] = df["ERROR_RATE_ML"].fillna(0)

    # ==========================================================
    # IKI + HOLD
    # ==========================================================
    df["IKI"] = df.groupby(
        ["PARTICIPANT_ID", "TEST_SECTION_ID"]
    )["PRESS_TIME"].diff().fillna(0).clip(lower=0)

    df["HOLD_TIME"] = (df["RELEASE_TIME"] - df["PRESS_TIME"]).clip(lower=0)

    # ==========================================================
    # KEYS
    # ==========================================================
    df["IS_BACKSPACE"] = df["KEYCODE"].isin([8, 46]).astype(int)

    total_keys = df.groupby(
        ["PARTICIPANT_ID", "TEST_SECTION_ID"]
    )["PRESS_TIME"].transform("count")

    # ==========================================================
    # KSPC FAMILY
    # ==========================================================
    sentence_len = df["SENTENCE"].str.len().replace(0, np.nan)

    df["KSPC"] = total_keys / (sentence_len + 1e-9)
    df["KSPC_WORD"] = total_keys / (df["SENTENCE"].str.split().str.len() + 1e-9)

    # ==========================================================
    # WORD FEATURES
    # ==========================================================
    words = df["SENTENCE"].str.split()

    df["WORD_COUNT"] = words.str.len()

    df["WORD_LENGTHS"] = words.apply(
        lambda x: [len(w) for w in x] if isinstance(x, list) else []
    )

    df["WORD_AVG_LEN"] = df["WORD_LENGTHS"].apply(lambda x: np.mean(x) if len(x) else 0)
    df["WORD_STD_LEN"] = df["WORD_LENGTHS"].apply(lambda x: np.std(x) if len(x) else 0)

    for i in range(1, 12):
        df[f"WORD_LEN_{i}"] = df["WORD_LENGTHS"].apply(
            lambda x: sum(1 for w in x if len(w) == i)
        )

    # ==========================================================
    # DIGRAPH FEATURES
    # ==========================================================
    df["DOUBLE_KEY"] = (df["KEYCODE"] == df["KEYCODE"].shift()).astype(int)

    df["DIGRAPH_TIME"] = df.groupby(
        ["PARTICIPANT_ID", "TEST_SECTION_ID"]
    )["PRESS_TIME"].diff().fillna(0).clip(lower=0)

    # ==========================================================
    # SESSION AGGREGATION
    # ==========================================================
    session = df.groupby(
        ["PARTICIPANT_ID", "TEST_SECTION_ID"]
    ).agg(

        # ---------------- IKI ----------------
        MEAN_IKI=("IKI", "mean"),
        STD_IKI=("IKI", "std"),
        MEDIAN_IKI=("IKI", "median"),
        MAX_IKI=("IKI", "max"),

        # ---------------- HOLD ----------------
        MEAN_HOLD=("HOLD_TIME", "mean"),

        # ---------------- KSPC ----------------
        MEAN_KSPC=("KSPC", "mean"),
        MEAN_KSPC_WORD=("KSPC_WORD", "mean"),

        # ---------------- WORD ----------------
        WORD_COUNT=("WORD_COUNT", "mean"),
        WORD_AVG_LEN=("WORD_AVG_LEN", "mean"),
        WORD_STD_LEN=("WORD_STD_LEN", "mean"),

        # ---------------- ERROR ----------------
        ERROR_RATE=("ERROR_RATE_ML", "mean"),

        # ---------------- CORRECTION ----------------
        BACKSPACE=("IS_BACKSPACE", "sum"),

        # ---------------- DIGRAPH ----------------
        DOUBLE_KEY_RATE=("DOUBLE_KEY", "mean"),
        MEAN_DIGRAPH=("DIGRAPH_TIME", "mean"),
        STD_DIGRAPH=("DIGRAPH_TIME", "std"),

        # ---------------- GLOBAL ----------------
        TOTAL_KEYS=("PRESS_TIME", "count"),
    )

    # ==========================================================
    # DERIVED FEATURES
    # ==========================================================
    session["IKI_CV"] = session["STD_IKI"] / (session["MEAN_IKI"] + 1e-9)

    session["IKI_RANGE"] = session["MAX_IKI"] - session["MEAN_IKI"]

    session["BACKSPACE_RATIO"] = session["BACKSPACE"] / (session["TOTAL_KEYS"] + 1e-9)

    session["DIGRAPH_CV"] = session["STD_DIGRAPH"] / (session["MEAN_DIGRAPH"] + 1e-9)

    session["KEY_ENTROPY"] = df.groupby(
        ["PARTICIPANT_ID", "TEST_SECTION_ID"]
    )["KEYCODE"].apply(entropy).values

    session["EFFORT_INDEX"] = (
        session["ERROR_RATE"] +
        session["BACKSPACE_RATIO"] +
        session["IKI_CV"]
    ) / 3

    session["COGNITIVE_LOAD_PROXY"] = (
        session["EFFORT_INDEX"] +
        session["MEAN_KSPC"]
    ) / 2

    return session.reset_index().fillna(0)