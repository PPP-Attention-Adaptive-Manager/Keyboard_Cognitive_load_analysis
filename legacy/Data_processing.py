import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

# ==========================================================
# CONFIG
# ==========================================================
N_JOBS = 12
print(f"🚀 Using {N_JOBS} CPU cores")

# ==========================================================
# FAST LEVENSHTEIN
# ==========================================================
def levenshtein(a, b):
    a, b = str(a), str(b)

    if a == b:
        return 0

    if abs(len(a) - len(b)) > 50:
        return max(len(a), len(b))

    m, n = len(a), len(b)
    if m == 0: return n
    if n == 0: return m

    dp = np.zeros((m + 1, n + 1), dtype=np.int16)

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        ca = a[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ca == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    return int(dp[m][n])

# ==========================================================
# ERROR FUNCTION
# ==========================================================
def compute_error(sentence, user_input, backspaces):
    sentence = str(sentence)
    user_input = str(user_input)

    if len(sentence) == 0 and len(user_input) == 0:
        return 0.0

    dist = levenshtein(sentence, user_input)

    lev_error = dist / max(len(sentence), 1)
    correction_penalty = backspaces / max(len(user_input), 1)

    return np.clip(0.7 * lev_error + 0.3 * correction_penalty, 0, 1)

# ==========================================================
# LOAD DATA
# ==========================================================
print("📥 Loading dataset...")
df = pd.read_csv("merged_first_sections.csv", low_memory=False)

# numeric safety
for c in ["PRESS_TIME", "RELEASE_TIME", "TEST_SECTION_ID"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["PRESS_TIME", "RELEASE_TIME", "TEST_SECTION_ID"])
df["TEST_SECTION_ID"] = df["TEST_SECTION_ID"].astype(int)

print(f"✔ Raw shape: {df.shape}")

# ==========================================================
# 🧠 SMART SAMPLING (IMPORTANT PART)
# ==========================================================
print("🧠 Selecting meaningful subset of data...")

session_quality = df.groupby(
    ["PARTICIPANT_ID", "TEST_SECTION_ID"]
).agg(
    rows=("PRESS_TIME", "count"),
    duration_ms=("PRESS_TIME", lambda x: x.max() - x.min()),
    error_proxy=("KEYCODE", lambda x: (x == 8).mean())
).reset_index()

# normalize duration
session_quality["duration_norm"] = session_quality["duration_ms"] / session_quality["duration_ms"].max()

# scoring function (balanced selection)
session_quality["quality_score"] = (
    0.5 * np.tanh(session_quality["rows"] / 200) +
    0.3 * session_quality["duration_norm"] +
    0.2 * (1 - session_quality["error_proxy"])
)

# keep top 60–75%
threshold = session_quality["quality_score"].quantile(0.25)

good_sessions = session_quality[
    session_quality["quality_score"] > threshold
][["PARTICIPANT_ID", "TEST_SECTION_ID"]]

df = df.merge(good_sessions, on=["PARTICIPANT_ID", "TEST_SECTION_ID"], how="inner")

print(f"✔ Reduced shape: {df.shape}")
# ==========================================================
# SESSION STATS
# ==========================================================
print("📊 Computing session stats...")

session_stats = df.groupby(["PARTICIPANT_ID", "TEST_SECTION_ID"]).agg(
    SESSION_START=("PRESS_TIME", "min"),
    SESSION_END=("RELEASE_TIME", "max"),
).reset_index()

session_stats["ELAPSED_TIME"] = (
    (session_stats["SESSION_END"] - session_stats["SESSION_START"])
    / 1000 / 60
).clip(lower=1e-6)

df = df.merge(session_stats, on=["PARTICIPANT_ID", "TEST_SECTION_ID"], how="left")

# ==========================================================
# TEMPORAL FEATURES
# ==========================================================
print("⏱ Temporal features...")

df["IKI"] = df.groupby(["PARTICIPANT_ID", "TEST_SECTION_ID"])["PRESS_TIME"].diff()
df["IKI"] = df["IKI"].fillna(0).clip(lower=0)

df["HOLD_TIME"] = (df["RELEASE_TIME"] - df["PRESS_TIME"]).clip(lower=0)

# ==========================================================
# BACKSPACE FLAG
# ==========================================================
print("⌨ Backspace detection...")

df["KEYCODE"] = pd.to_numeric(df["KEYCODE"], errors="coerce")
df["IS_BACKSPACE"] = df["KEYCODE"].isin([8, 46]).astype(int)

df["SENTENCE"] = df["SENTENCE"].fillna("").astype(str)
df["USER_INPUT"] = df["USER_INPUT"].fillna("").astype(str)

df["SENTENCE_LENGTH"] = df["SENTENCE"].str.len()

# ==========================================================
# PARALLEL ERROR (12 cores)
# ==========================================================
print("⚡ Computing error (parallel)...")

groups = list(df.groupby(["PARTICIPANT_ID", "TEST_SECTION_ID", "SENTENCE"]))

def process_group(item):
    (pid, sid, sent), g = item
    err = compute_error(
        sent,
        g["USER_INPUT"].iloc[-1],
        g["IS_BACKSPACE"].sum()
    )
    return pid, sid, sent, err

results = Parallel(n_jobs=N_JOBS)(
    delayed(process_group)(g) for g in tqdm(groups)
)

sentence_error = pd.DataFrame(
    results,
    columns=["PARTICIPANT_ID", "TEST_SECTION_ID", "SENTENCE", "ERROR_RATE"]
)

df = df.merge(sentence_error,
              on=["PARTICIPANT_ID", "TEST_SECTION_ID", "SENTENCE"],
              how="left")

# ==========================================================
# SESSION FEATURES
# ==========================================================
print("📊 Session feature engineering...")

session_features = df.groupby(["PARTICIPANT_ID", "TEST_SECTION_ID"]).agg(
    MEAN_IKI=("IKI", "mean"),
    MEDIAN_IKI=("IKI", "median"),
    STD_IKI=("IKI", "std"),

    IKI_CV=("IKI", lambda x: x.std() / (x.mean() + 1e-9)),

    LONG_PAUSE_COUNT=("IKI", lambda x: (x > 1000).sum()),
    PAUSE_RATIO=("IKI", lambda x: (x > 1000).mean()),

    MEAN_HOLD_TIME=("HOLD_TIME", "mean"),
    MAX_HOLD_TIME=("HOLD_TIME", "max"),

    TOTAL_KEYSTROKES=("LETTER", "count"),
    BACKSPACE_COUNT=("IS_BACKSPACE", "sum"),

    MEAN_ERROR_RATE=("ERROR_RATE", "mean"),
    MEAN_SENTENCE_LENGTH=("SENTENCE_LENGTH", "mean"),
).reset_index()

# ==========================================================
# FINAL FEATURES
# ==========================================================
print("🧠 Final features...")

session_features = session_features.merge(
    session_stats[["PARTICIPANT_ID", "TEST_SECTION_ID", "ELAPSED_TIME"]],
    on=["PARTICIPANT_ID", "TEST_SECTION_ID"],
    how="left"
)

session_features["COGNITIVE_WPM"] = (
    (session_features["MEAN_SENTENCE_LENGTH"] / 5)
    / (session_features["ELAPSED_TIME"] + 1e-9)
)

final_lengths = df.groupby(["PARTICIPANT_ID", "TEST_SECTION_ID"])["USER_INPUT"] \
    .last().str.len().reset_index(name="FINAL_TEXT_LENGTH")

session_features = session_features.merge(
    final_lengths,
    on=["PARTICIPANT_ID", "TEST_SECTION_ID"],
    how="left"
)

session_features["KSPC"] = (
    session_features["TOTAL_KEYSTROKES"]
    / (session_features["FINAL_TEXT_LENGTH"] + 1e-9)
)
session_features["BACKSPACE_RATIO"] = (
    session_features["BACKSPACE_COUNT"] /
    (session_features["TOTAL_KEYSTROKES"] + 1e-9)
)


session_features = session_features.fillna(0)

# ==========================================================
# SAVE
# ==========================================================
print("💾 Saving...")
session_features.to_csv("session_cogload_metrics.csv", index=False)

print("✅ DONE (SMART SAMPLING + 12 CORE + STABLE PIPELINE)")