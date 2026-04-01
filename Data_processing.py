import pandas as pd
import numpy as np

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("merged_first_sections.csv")
df["RELEASE_TIME"] = pd.to_numeric(df["RELEASE_TIME"], errors="coerce")
df = df.dropna(subset=["RELEASE_TIME", "PRESS_TIME"])

print(df.info())
# -----------------------------
# Session timing
# -----------------------------
session_stats = df.groupby(["PARTICIPANT_ID", "TEST_SECTION_ID"]).agg(
    SESSION_START=("PRESS_TIME", "min"),
    SESSION_END=("RELEASE_TIME", "max")
).reset_index()

session_stats["ELAPSED_TIME"] = (
    session_stats["SESSION_END"] - session_stats["SESSION_START"]
) / 1000 / 60  # minutes

# Merge back
df = df.merge(session_stats, on=["PARTICIPANT_ID", "TEST_SECTION_ID"], how="left")

# -----------------------------
# Sort for temporal features
# -----------------------------
df = df.sort_values(['PARTICIPANT_ID', 'TEST_SECTION_ID', 'PRESS_TIME'])

# -----------------------------
# IKI (per session now)
# -----------------------------
df['IKI'] = df.groupby(['PARTICIPANT_ID', 'TEST_SECTION_ID'])['PRESS_TIME'].diff()
df['IKI'] = df['IKI'].fillna(0)
df.loc[df['IKI'] < 0, 'IKI'] = 0

# -----------------------------
# Hold time
# -----------------------------
df['HOLD_TIME'] = df['RELEASE_TIME'] - df['PRESS_TIME']

# -----------------------------
# Sentence-level helpers
# -----------------------------
df['SENTENCE_LENGTH'] = df['SENTENCE'].str.len()
df['IS_BACKSPACE'] = df['LETTER'].str.upper().isin(['BACKSPACE', 'BKSP', 'DELETE', 'DEL']).astype(int)
# -----------------------------
# Error rate (per sentence → then session avg)
# -----------------------------
def compute_error_rate(sentence, user_input):
    errors = sum(a != b for a, b in zip(str(sentence), str(user_input)))
    return errors / max(len(str(sentence)), 1)

sentence_error = df.groupby(['PARTICIPANT_ID','TEST_SECTION_ID','SENTENCE']).agg(
    ERROR_RATE=('USER_INPUT', lambda x: compute_error_rate(
        df.loc[x.index, 'SENTENCE'].iloc[0],
        x.iloc[0]
    ))
).reset_index()

# -----------------------------
# Merge sentence error back
# -----------------------------
df = df.merge(sentence_error, on=['PARTICIPANT_ID','TEST_SECTION_ID','SENTENCE'], how='left')

# -----------------------------
# 🚀 SESSION-LEVEL FEATURES
# -----------------------------
session_features = df.groupby(['PARTICIPANT_ID','TEST_SECTION_ID']).agg(

    # Timing
    MEAN_IKI=('IKI','mean'),
    MEDIAN_IKI=('IKI','median'),
    STD_IKI=('IKI','std'),

    # Advanced variability
    IKI_CV=('IKI', lambda x: x.std() / x.mean() if x.mean() > 0 else 0),

    # Pause behavior (important cognitive signal)
    LONG_PAUSE_COUNT=('IKI', lambda x: (x > 1000).sum()),  # >1s pauses
    PAUSE_RATIO=('IKI', lambda x: (x > 1000).sum() / len(x)),

    # Hold time
    MEAN_HOLD_TIME=('HOLD_TIME','mean'),
    MAX_HOLD_TIME=('HOLD_TIME','max'),

    # Typing behavior
    TOTAL_KEYSTROKES=('LETTER','count'),
    BACKSPACE_COUNT=('IS_BACKSPACE','sum'),

    # Error
    MEAN_ERROR_RATE=('ERROR_RATE','mean'),

    # Sentence info
    MEAN_SENTENCE_LENGTH=('SENTENCE_LENGTH','mean')

).reset_index()

# -----------------------------
# Add session duration
# -----------------------------
session_features = session_features.merge(
    session_stats[['PARTICIPANT_ID','TEST_SECTION_ID','ELAPSED_TIME']],
    on=['PARTICIPANT_ID','TEST_SECTION_ID'],
    how='left'
)

# -----------------------------
# Compute WPM per session
# -----------------------------
session_features['WPM'] = (
    (session_features['MEAN_SENTENCE_LENGTH'] / 5)
    / session_features['ELAPSED_TIME']
)

# -----------------------------
# Compute KSPC
# -----------------------------
session_features['KSPC'] = (
    session_features['TOTAL_KEYSTROKES'] /
    session_features['MEAN_SENTENCE_LENGTH']
)

# -----------------------------
# Save
# -----------------------------
session_features.to_csv("session_cogload_metrics.csv", index=False)

print("✅ Saved session-level cognitive load features to 'session_cogload_metrics.csv'")
print(session_features.head())