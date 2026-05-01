# Keyboard Cognitive Load Analysis (V2 current, V3 experimental)

This repo explores multiple generations of a system that infers cognitive signals from typing / keystroke dynamics.

- V2 is the “current” working system: feature engineering → proxy targets → model(s) → real-time dashboard.
- V3 is an experimental next-gen pipeline (parquet + embeddings).
- legacy/ contains older prototypes and one-off scripts.

---

## High-level Architecture (V2)

```
Raw keyboard events / keystroke logs
                                ↓
Temporal features (IKI, hold time)
                                ↓
Session feature engineering (aggregations + error/correction metrics)
                                ↓
Proxy layer (rule-based targets: cogload / stress / unfocus)
                                ↓
Model layer (PyTorch BehaviorNet regression)  +  optional sklearn baselines
                                ↓
Real-time inference (RealtimeFeatureBuilder + CognitiveModel)
                                ↓
UI (Tkinter dashboard)
```

---

## Important Note About Paths / Working Directory

Some scripts in V2 were written assuming you run them from inside the V2 folder (because they use relative paths like `data/...` and `models/...`).

Rule of thumb:

- If a script reads/writes `data/...` or `models/...` (without `V2/` in the path), run it with your current working directory set to `V2/`.
- If a script is run as a Python module (e.g. `python -m V2.ui.dashboard`), it can usually be run from the repo root.

---

## “How do I run it?” (Common Workflows)

### Workflow A — Build V2 session features + proxies

Goal: produce `V2/data/processed/session_features.csv` which includes engineered features + proxy targets.

1) Ensure you have a raw merged keystroke file at:

- `V2/data/raw/merged_first_sections.csv`

Expected columns used by the pipeline include (at minimum):

- `PARTICIPANT_ID`, `TEST_SECTION_ID`, `PRESS_TIME`, `RELEASE_TIME`
- plus sentence columns if you want error features: `SENTENCE`, `USER_INPUT`, `KEYCODE`

2) Run the feature pipeline (run from inside `V2/`):

```bash
cd V2
python features/run_feature_pipeline.py
```

What it does:

- Step 1: raw → temporal (`temporal_sequences.csv`)
- Step 2: temporal → engineered features (`session_features_nocog.csv`)
- Step 3: features → proxies (`session_features.csv`)

### Workflow B — Train the V2 proxy-regression model (BehaviorNet)

Goal: train a small PyTorch network that predicts the proxy targets from a small feature set.

```bash
cd V2
python training/train_behavior_model.py
```

Output:

- `V2/models/behavior_model.pth`

### Workflow C — Run the real-time dashboard (V2)

Goal: open a Tkinter UI; as you type, it computes live features and predicts:

- `cognitive_load`, `stress`, `unfocus`

Run from repo root:

```bash
python -m V2.ui.dashboard
```

Prerequisite:

- `V2/models/behavior_model.pth` must exist (train it via Workflow B)

### Workflow D — Student labeling + baseline model training (V2/training)

This is a separate training pipeline that labels keystrokes with a pre-trained “student” encoder + clustering, then trains sklearn models for classification/regression.

1) Generate a labeled dataset (runs from repo root):

```bash
python -m V2.preprocessing.labeling_with_student
```

Outputs:

- `V2/fully_labeled_cognitive_dataset.csv`

2) Train/evaluate baseline models (runs from repo root):

```bash
python -m V2.training.keystrokes_model
```

Or via the root wrapper:

```bash
python keystrokes_model.py
```

### Workflow E — V3 experimental pipelines

V3 is not wired into V2. It contains:

- Phase 1: stream raw `.txt` keystrokes → parquet with sequences + engineered features
- Phase 2: train a contrastive embedding model and export `.npy` embeddings

```bash
python V3/Phase_1/pipeline.py
python V3/phase_2/pipeline.py
python V3/phase_2/eval.py
```

Notes:

- The V3 scripts currently contain hard-coded absolute paths (Windows-style) for input/output. If you run them on a different machine, update the `DATA_DIR` / `DATA_PATH` / `OUTPUT_DIR` constants inside those scripts.
- `V3/phase_2/pipeline.py` saves `train_embeddings.npy` and `test_embeddings.npy` to your current working directory. If you want them under `V3/`, run the script from inside the `V3/` folder.

---

## Repository Map (File-by-file)

### Root

- `Readme.md`
        - This document.

- `keystrokes_model.py`
        - Convenience wrapper that calls `V2/training/keystrokes_model.py`.
        - Useful when you just want to run the sklearn baseline training/eval from repo root.

---

### legacy/ (older prototypes; not used by V2 by default)

These scripts are mostly proof-of-concept / exploratory. They are helpful references but are not the current pipeline.

- `legacy/Data-merging.py`
        - Parallel merges raw keystroke `.txt` files into `merged_first_sections.csv`.
        - Filters to the first `N_SECTIONS` test sections per file.

- `legacy/Data_processing.py`
        - Early feature extraction from `merged_first_sections.csv`.
        - Produces session-level aggregates and saves `session_cogload_metrics.csv`.

- `legacy/Model_training.py`
        - Trains a RandomForestRegressor on `session_cogload_metrics.csv`.
        - Creates a synthetic cognitive-load proxy formula as the training target.
        - Saves `rf_cogload_model_complex.pkl` + `feature_scaler_complex.pkl`.

- `legacy/testing_model.py`
        - Tkinter GUI prototype that uses the legacy RandomForest model and a manual proxy.
        - Uses a 60s sliding window of key timings.

- `legacy/user_profile.py`
        - Persistent JSON-based user profiling class (`UserProfile`) with EMA updates.
        - Tracks avg/variance for IKI/pause/error and can normalize values.

- `legacy/dqn_agent.py`
        - Standalone DQN agent implementation (Q-network + replay buffer).
        - Not integrated with the V2 RL scaffolding.

- `legacy/data_understanding.py`
        - One-off exploratory script (loads first 1,000,000 rows and prints stats).

- `legacy/requirements.txt`
        - Minimal dependency list used by the legacy scripts.

---

### V2/ (current system)

#### V2/data/

- `V2/data/raw/`
        - Expected location for the merged raw CSV used by the feature pipeline.

- `V2/data/processed/`
        - Pipeline outputs (caches + final datasets).

#### V2/preprocessing/

- `V2/preprocessing/build_sessions.py`
        - Parallel loads raw `.txt` keystroke files and builds session segmentation.
        - Output: `V2/data/processed/sessions_raw.csv`.

- `V2/preprocessing/temporal_builder.py`
        - Adds temporal columns to event-level data:
                - `IKI` (inter-key interval)
                - `HOLD_TIME` (release - press)

- `V2/preprocessing/labeling_with_student.py`
        - Uses a pre-trained LSTM “student” encoder (`V2/student_model.pt`) to create embeddings(the encoder is in ).
        - Fits MiniBatchKMeans (3 clusters) and writes:
                - `cognitive_label` (cluster id)
                - `cognitive_risk` (distance-based risk score)
        - Output: `V2/fully_labeled_cognitive_dataset.csv`.

- `V2/preprocessing/inspect_student_ckpt.py`
        - Diagnostic script to inspect the student checkpoint structure and tensor shapes.

- `V2/preprocessing/data_test.py`
        - Quick schema/dtype print for `fully_labeled_cognitive_dataset.csv`.

#### V2/features/

- `V2/features/feature_engineering.py`
        - The main session feature builder used by the pipeline.
        - Highlights:
                - Fast Levenshtein distance via Numba (`levenshtein_fast`) to estimate error.
                - Session aggregation: IKI stats, hold stats, correction/backspace ratio, entropy.
        - Output: a session-level feature DataFrame.

- `V2/features/run_feature_pipeline.py`
        - Orchestrates the 3-stage pipeline with caching:
                1) temporal sequences
                2) session features
                3) proxy targets
        - Output: `V2/data/processed/session_features.csv`.

#### V2/proxies/

- `V2/proxies/proxy_definitions.py`
        - Computes rule-based proxy targets per participant:
                - `COGLOAD_PROXY`, `STRESS_PROXY`, `UNFOCUS_PROXY`
        - Does z-scoring per participant, smoothing, then min-max scaling.

#### V2/training/

- `V2/training/train_behavior_model.py`
        - Trains a small PyTorch MLP on `V2/data/processed/session_features.csv`.
        - Predicts: `COGLOAD_PROXY`, `STRESS_PROXY`, `UNFOCUS_PROXY`.
        - Saves: `V2/models/behavior_model.pth` with feature metadata.

- `V2/training/keystrokes_model.py`
        - Trains/evaluates sklearn baselines on `V2/fully_labeled_cognitive_dataset.csv`:
                - classification: `cognitive_label`
                - regression: `cognitive_risk`
        - Uses group-safe train/test splitting by `PARTICIPANT_ID`.

#### V2/models/

- `V2/models/behavior_net.py`
        - Defines the PyTorch model architecture (`BehaviorNet`) used by V2.

- `V2/models/cognitive_model.py`
        - Inference wrapper (`CognitiveModel`) around `BehaviorNet`.
        - Handles:
                - sliding window aggregation
                - per-user normalization (via `V2/personalization/user_profile.py`)
                - clipping outputs to [0, 1]

- `V2/models/compare_models.py`
        - Utility script to compare a trained NN (`behavior_model.pth`) vs a RandomForest (`behavior_model.pkl`).
        - Computes aggregate metrics and multiple “importance” estimates.

- Model artifacts in this folder:
        - `behavior_model.pth`: trained BehaviorNet checkpoint (used by the dashboard)
        - `cogload_label_model.joblib`, `cogload_risk_model.joblib`: joblib models used by `V2/realtime/realtime_analyzer.py`

#### V2/inference/

- `V2/inference/realtime_features.py`
        - Real-time feature builder (`RealtimeFeatureBuilder`).
        - Consumes key press/release events and builds a single-row feature DataFrame.

#### V2/personalization/

- `V2/personalization/user_profile.py`
        - Lightweight in-memory `UserProfile` used by `CognitiveModel` for per-user mean/std normalization.

#### V2/ui/

- `V2/ui/dashboard.py`
        - Main Tkinter dashboard.
        - Integrates:
                - `RealtimeFeatureBuilder`
                - `UserProfile`
                - `CognitiveModel`
        - Displays smoothed predictions and a live Matplotlib chart.

#### V2/realtime/ (older real-time prototype)

- `V2/realtime/realtime_analyzer.py`
        - Prototype real-time analyzer using joblib models (`cogload_label_model.joblib`, `cogload_risk_model.joblib`).
        - Uses `pynput` to listen globally for keyboard events.

#### V2/rl/ (experimental)

- `V2/rl/behavior_rl_env.py`
        - Simple RL-style environment for learning weights over signals (ML / proxies / user memory).

- `V2/rl/simple_policy.py`
        - Very small adaptive policy stub that perturbs weights based on reward sign.

#### V2/utils/

- `V2/utils/path.py`
        - Path helpers used to locate `V2/data` and `V2/models` relative to the repo root.

---

### V3/ (experimental embedding-based approach)

- `V3/Phase_1/pipeline.py`
        - Streams raw `.txt` keystroke logs, builds sequences, computes engineered features, and writes parquet batches.
        - Output: `V3/processed/phase1_final.parquet`.

- `V3/phase_2/pipeline.py`
        - Loads the parquet from phase 1, trains a sequence+feature embedding model with a contrastive objective.
        - Outputs: `train_embeddings.npy`, `test_embeddings.npy` (saved to the current working directory).

- `V3/phase_2/eval.py`
        - Embedding evaluation utilities:
                - cosine similarity
                - self-retrieval accuracy
                - UMAP visualization

---

## Suggested “Start Here” for New Users

If your goal is to see the system working end-to-end with the least moving parts:

1) Create/confirm `V2/data/raw/merged_first_sections.csv`
2) `cd V2` → run `python features/run_feature_pipeline.py`
3) `cd V2` → run `python training/train_behavior_model.py`
4) from repo root → run `python -m V2.ui.dashboard`

## Experimentation folder
that folder has a full experimentation of a dataset merged with results and all most are negative