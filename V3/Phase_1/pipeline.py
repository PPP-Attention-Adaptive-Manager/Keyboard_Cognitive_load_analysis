import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import multiprocessing as mp
from tqdm import tqdm
import torch

# =========================
# CONFIG
# =========================
DATA_DIR = Path(r"D:\files\PPP\Dataset\Keystrokes\files")

OUTPUT_DIR = Path(
    r"D:\files\PPP\github repo (linux)\Keyboard_Cognitive_load_analysis\V3\processed"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_PATH = OUTPUT_DIR / "phase1_final.parquet"

# ---------- Sampling ----------
MAX_FILES = 20000            # None = all files
SESSIONS_PER_FILE = 5

# ---------- Dataset ----------
MIN_SESSION_LEN = 15
IKL_THRESHOLD = 250

# ---------- Performance ----------
BATCH_SIZE = 5000
WORKERS = max(1, mp.cpu_count() - 2)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("DEVICE:", DEVICE)

# =========================
# SAFE INT
# =========================
def safe_int(x):
    try:
        if x is None:
            return None
        x = str(x).strip()
        if x == "" or x.lower() in ["na", "null", "none"]:
            return None
        return int(float(x))
    except:
        return None


# =========================
# STREAM FILE
# =========================
def stream_file(file_path):

    encodings = ["utf-8", "latin-1", "cp1252"]

    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc, errors="strict") as f:

                header = f.readline().strip().split("\t")

                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) != len(header):
                        continue

                    row = dict(zip(header, parts))

                    press = safe_int(row.get("PRESS_TIME"))
                    release = safe_int(row.get("RELEASE_TIME"))
                    code = safe_int(row.get("KEYCODE"))

                    if press is None or release is None or code is None:
                        continue

                    yield {
                        "user_id": int(row["PARTICIPANT_ID"]),
                        "session_id": int(row["TEST_SECTION_ID"]),
                        "code": code,
                        "press": press,
                        "release": release,
                    }
            return
        except Exception:
            continue

    # fallback
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline().strip().split("\t")

        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != len(header):
                continue

            row = dict(zip(header, parts))

            press = safe_int(row.get("PRESS_TIME"))
            release = safe_int(row.get("RELEASE_TIME"))
            code = safe_int(row.get("KEYCODE"))

            if press is None or release is None or code is None:
                continue

            yield {
                "user_id": int(row["PARTICIPANT_ID"]),
                "session_id": int(row["TEST_SECTION_ID"]),
                "code": code,
                "press": press,
                "release": release,
            }


# =========================
# BUILD SEQUENCE
# =========================
def build_sequence(events):

    events.sort(key=lambda x: x["press"])

    seq = []
    prev_release = None

    for e in events:
        hold = e["release"] - e["press"]
        ikl = 0 if prev_release is None else e["press"] - prev_release
        prev_release = e["release"]

        seq.append(
            {
                "code": e["code"],
                "hold": hold,
                "ikl": ikl,
                "press": e["press"],
                "release": e["release"],
            }
        )

    return seq


# =========================
# VALIDATION
# =========================
def is_valid(seq):

    if len(seq) < MIN_SESSION_LEN:
        return False

    holds = [x["hold"] for x in seq]

    if max(holds) > 15000:
        return False

    if any(h < 0 for h in holds):
        return False

    return True


# =========================
# GPU FEATURE ENGINE
# =========================
def compute_features_gpu(seq):

    holds = torch.tensor(
        [x["hold"] for x in seq],
        dtype=torch.float32,
        device=DEVICE,
    )

    ikl = torch.tensor(
        [x["ikl"] for x in seq],
        dtype=torch.float32,
        device=DEVICE,
    )

    codes = torch.tensor(
        [x["code"] for x in seq],
        device=DEVICE,
    )

    error_rate = (codes == 8).float().mean()

    ikl_mean = ikl.mean()
    ikl_std = ikl.std()

    spikes = (ikl > (ikl_mean + 2 * ikl_std)).float().mean()

    bursts = (ikl < IKL_THRESHOLD).float()

    pause_ratio = (ikl > 1000).float().mean()

    mid = len(seq) // 2

    def speed(part):
        duration = part[-1]["release"] - part[0]["press"] + 1
        return len(part) / (duration / 1000)

    decay = speed(seq[:mid]) - speed(seq[mid:])

    return {
        "hold_mean": holds.mean().item(),
        "hold_std": holds.std().item(),
        "ikl_mean": ikl_mean.item(),
        "ikl_std": ikl_std.item(),
        "ikl_spike_ratio": spikes.item(),
        "error_rate": error_rate.item(),
        "burst_mean": bursts.mean().item(),
        "burst_max": int(bursts.sum().item()),
        "pause_ratio": pause_ratio.item(),
        "consistency_decay": float(decay),
    }


# =========================
# PROCESS FILE
# =========================
def process_file(file_path):

    sessions = defaultdict(list)
    session_order = []

    for row in stream_file(file_path):

        key = (row["user_id"], row["session_id"])

        if key not in sessions:
            session_order.append(key)

            if len(session_order) > SESSIONS_PER_FILE:
                break

        sessions[key].append(row)

    results = []

    for key in session_order:

        seq = build_sequence(sessions[key])

        if not is_valid(seq):
            continue

        features = compute_features_gpu(seq)

        results.append(
            {
                "user_id": key[0],
                "session_id": key[1],
                "sequence": json.dumps(seq),
                **{f"feat_{k}": v for k, v in features.items()},
            }
        )

    return results


# =========================
# PARQUET BATCH WRITER
# =========================
def write_batch(rows, first_write=False):

    if not rows:
        return

    df = pd.DataFrame(rows)

    if first_write or not PARQUET_PATH.exists():
        df.to_parquet(PARQUET_PATH, index=False)
    else:
        old = pd.read_parquet(PARQUET_PATH)
        df = pd.concat([old, df], ignore_index=True)
        df.to_parquet(PARQUET_PATH, index=False)


# =========================
# MAIN PIPELINE
# =========================
def run_pipeline():

    files = list(DATA_DIR.rglob("*.txt"))

    if MAX_FILES is not None:
        np.random.seed(42)
        files = np.random.choice(files, MAX_FILES, replace=False).tolist()

    print("Workers:", WORKERS)
    print("Files:", len(files))

    buffer = []
    first_write = True
    total_sessions = 0

    with mp.Pool(WORKERS) as pool:

        for result in tqdm(
            pool.imap_unordered(process_file, files),
            total=len(files),
        ):

            buffer.extend(result)
            total_sessions += len(result)

            if len(buffer) >= BATCH_SIZE:
                write_batch(buffer, first_write)
                first_write = False
                buffer.clear()

    write_batch(buffer, first_write)

    print("Total sessions saved:", total_sessions)


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    mp.freeze_support()

    run_pipeline()

    print("✅ PHASE 1 COMPLETE")