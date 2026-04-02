# Keyboard Cognitive Load Analysis Pipeline

## Overview

The **Keyboard Cognitive Load Analysis Pipeline** is a comprehensive machine learning system designed to quantify cognitive load in real-time by analyzing typing behavior patterns. This project processes keystroke dynamics from user interactions, extracts behavioral features, trains a predictive model, and provides real-time cognitive load assessment through an interactive GUI.

The system uses a hybrid approach combining machine learning (Random Forest Regressor) with a mathematically-derived psychometric proxy to create a robust cognitive load indicator (0-10 scale).

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Workflow Overview](#workflow-overview)
- [Detailed Pipeline Steps](#detailed-pipeline-steps)
- [Usage Guide](#usage-guide)
- [Cognitive Load Thresholds](#cognitive-load-thresholds)
- [Output Files](#output-files)
- [Troubleshooting](#troubleshooting)
- [Technical Details](#technical-details)

---

## Features

✅ **Multi-stage Data Pipeline**: Seamlessly integrates raw keystroke logs from multiple sources  
✅ **Advanced Feature Engineering**: Extracts 15+ behavioral metrics from keystroke dynamics  
✅ **Machine Learning Model**: Random Forest Regressor trained on 16M+ keystroke events  
✅ **Real-time Monitoring**: Live GUI for cognitive load assessment during active typing  
✅ **Hybrid Prediction**: Combines ML predictions with mathematical proxy formulas  
✅ **Data Validation**: Comprehensive data integrity checks before processing  

---

## Project Structure

```
Keyboard_Cognitive_load_analysis/
├── Readme.md                          # This file - comprehensive documentation
├── requirements.txt                   # Python dependencies
├── Data-merging.py                    # Stage 1: Consolidate raw logs
├── data_understanding.py              # Stage 2: Verify data integrity
├── Data_processing.py                 # Stage 3: Feature engineering
├── Model_training.py                  # Stage 4: Train ML model
├── testing_model.py                   # Stage 5: Real-time deployment & GUI
└── [outputs]/
    ├── merged_first_sections.csv      # Merged raw keystroke data
    ├── session_cogload_metrics.csv    # Engineered features (session-level)
    ├── rf_cogload_model_complex.pkl   # Trained Random Forest model
    └── feature_scaler_complex.pkl     # Feature scaling parameters
```

---

## Installation

### Prerequisites
- **Python 3.7+** (recommend 3.9 or higher)
- **pip** package manager

### Setup

1. **Clone or download the project:**
   ```bash
   cd Keyboard_Cognitive_load_analysis
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python -c "import pandas, sklearn, matplotlib, seaborn, pynput, tkinter; print('All dependencies installed successfully!')"
   ```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `pandas` | Data manipulation, merging, and analysis |
| `scikit-learn` | Machine learning (Random Forest, StandardScaler) |
| `matplotlib` & `seaborn` | Data visualization and exploratory analysis |
| `pynput` | Keystroke monitoring (live mode) |
| `tkinter` | GUI framework for real-time monitoring |
| `tqdm` | Progress bars for long-running processes |

---

## Workflow Overview

The pipeline follows a sequential 5-stage process:

```
Raw Keystroke Data → Merge → Validate → Process Features → Train Model → Real-time Testing
   (Stage 1)        (Stage 2)  (Stage 3)    (Stage 4)        (Stage 5)
```

---

## Detailed Pipeline Steps

### **Stage 1: Data Merging** (`Data-merging.py`)

**Goal**: Consolidate fragmented raw keystroke logs from multiple participants/sessions into a unified dataset.

**Input Requirements**:
- Multiple CSV or TXT files containing keystroke records
- Files should contain columns like: `PARTICIPANT_ID`, `TEST_SECTION_ID`, `PRESS_TIME`, `RELEASE_TIME`, `LETTER`, `KEYCODE`

**Processing**:
- Standardizes column headers across all input files
- Ensures consistent timestamp formats (PRESS_TIME, RELEASE_TIME)
- Validates column alignment
- Concatenates all files into a single dataframe
- Removes duplicates and sorts by timestamp

**Output**: 
- `merged_first_sections.csv` - Master dataset containing all merged keystroke events

**Usage**:
```bash
python Data-merging.py
```

---

### **Stage 2: Data Understanding** (`data_understanding.py`)

**Goal**: Verify data integrity and understand how special characters (especially backspaces) are encoded.

**Key Validations**:
- Scans `LETTER` and `KEYCODE` columns for encoding patterns
- Identifies how backspace/deletion events are recorded:
  - ASCII representation (e.g., ASCII code 8)
  - String labels (e.g., 'Backspace', 'BKSP', 'Delete')
  - Custom encoding schemes

**Critical Note**:
⚠️ If backspace encoding is not correctly identified, the `BACKSPACE_RATIO` feature will remain zero during model training, potentially reducing prediction accuracy.

**Output**:
- Console report of data encoding and validation results
- Summary statistics of problematic records (if any)

**Usage**:
```bash
python data_understanding.py
```

---

### **Stage 3: Data Processing** (`Data_processing.py`)

**Goal**: Transform raw event-level keystroke data (16M+ individual key presses) into aggregated session-level behavioral features.

**Input**: `merged_first_sections.csv`

**Processing Steps**:
1. Groups keystroke events by `PARTICIPANT_ID` and `TEST_SECTION_ID`
2. Aggregates individual key press events into session-level metrics
3. Calculates 15+ behavioral features from keystroke dynamics

**Key Engineered Features**:

| Feature | Description | Interpretation |
|---------|-------------|-----------------|
| **IKI_CV** | Coefficient of Variation of Inter-Keystroke Intervals | Measures rhythm consistency; higher values indicate erratic typing |
| **PAUSE_RATIO** | Total pause time ÷ total session duration (%) | Higher ratios indicate more thinking/hesitation |
| **PAUSE_PER_MIN** | Number of pauses per minute | Frequency of hesitations during the session |
| **BACKSPACE_RATIO** | Backspace count ÷ total key count (%) | Error correction frequency; indicates uncertainty |
| **AVG_HOLD_TIME** | Average key hold duration (ms) | Typing deliberation; higher=more careful/tense typing |
| **MAX_HOLD_TIME** | Maximum key hold duration (ms) | Peak deliberation moment |
| **TYPING_SPEED** | Keys per second | Overall typing velocity |
| **ERROR_CORRECTION_RATE** | Errors detected and corrected per minute | Problem-solving activity |

**Output**: 
- `session_cogload_metrics.csv` - One row per session with all engineered features

**Usage**:
```bash
python Data_processing.py
```

---

### **Stage 4: Model Training** (`Model_training.py`)

**Goal**: Train a machine learning model to predict cognitive load from behavioral features.

**Target Variable (Proxy Formula)**:
The target is a complex, non-linear mathematical formula that encodes cognitive load principles:

$$\text{CogLoad} = 0.3 \cdot \ln(1 + \text{Hold} \cdot \text{Pause}) + 0.2 \cdot \text{Max\_Hold} + 0.2 \cdot \text{Pause\_Per\_Min}^{1.5} + \ldots$$

This formula combines multiple typing metrics into a single cognitive load score (range: 0-10).

**Model Architecture**:
- **Algorithm**: Random Forest Regressor
- **Trees**: 200 decision trees
- **Feature Scaling**: StandardScaler (zero mean, unit variance)
- **Train-Test Split**: 80-20 split with shuffled random state

**Training Process**:
1. Load features from `session_cogload_metrics.csv`
2. Compute target variable using the proxy formula
3. Scale features using StandardScaler for optimal RF performance
4. Train Random Forest Regressor on 80% of data
5. Evaluate on 20% holdout test set

**Evaluation Metrics**:
- **R² Score**: Proportion of variance explained (0-1, higher is better)
- **Mean Squared Error (MSE)**: Average squared prediction error
- **Root Mean Squared Error (RMSE)**: Error in original units
- **Mean Absolute Error (MAE)**: Average absolute prediction error

**Outputs**:
- `rf_cogload_model_complex.pkl` - Serialized trained Random Forest model
- `feature_scaler_complex.pkl` - Fitted StandardScaler for feature normalization
- Training metrics and performance report (printed to console)

**Usage**:
```bash
python Model_training.py
```

**Example Output**:
```
Model Performance:
R² Score: 0.887
RMSE: 0.542
MAE: 0.398
```

---

### **Stage 5: Real-time Testing & GUI** (`testing_model.py`)

**Goal**: Deploy the trained model in a live monitoring environment with an interactive GUI for real-time cognitive load assessment during actual keyboard usage.

**Features**:

#### **Sliding Window Analysis** (60-second window)
- Only analyzes keystrokes from the last 60 seconds
- Continuously updates as new keystrokes are recorded
- Ignores historical data, focusing on current state

#### **Frontier Logic** (Pause Classification)
Differentiates between two mental states using a 5-second boundary:

- **Pauses 1-5 seconds**: Classified as **Hesitation**
  - Indicates active problem-solving/uncertainty
  - Increases cognitive load score
  
- **Pauses > 5 seconds**: Classified as **Reflection**
  - Indicates thoughtful pause or task transition
  - Stabilizes or reduces cognitive load score

#### **Hybrid Prediction** (Ensemble approach)
Final cognitive load combines two methods:

$$\text{Final Load} = \frac{\text{ML Prediction} + \text{Proxy Formula}}{2}$$

- **ML Prediction**: Random Forest output (learned complex patterns)
- **Proxy Formula**: Mathematical formula (explicit cognitive principles)
- **Result**: Robust, interpretable cognitive load estimate

#### **Interactive GUI**
- Real-time monitoring display
- Current cognitive load indicator
- Live typing metrics visualization
- Session history
- Status updates and alerts

**Usage**:
```bash
python testing_model.py
```

---

## Cognitive Load Thresholds

Cognitive load is measured on a **0-10 scale**:

| Range | Category | Characteristics | User State |
|-------|----------|-----------------|------------|
| **0.0 - 3.5** | 🟢 **Low** | Smooth, rhythmic typing; minimal pauses; few errors; consistent inter-keystroke intervals | Active flow state; user is comfortable |
| **3.5 - 7.0** | 🟡 **Moderate** | Choppy typing; increased hesitations; more backspaces; variable rhythm | User thinking; increased task difficulty |
| **7.0 - 10.0** | 🔴 **High** | Significant error rates; long pauses; frequent corrections; sporadic typing pattern | High cognitive stress; potential cognitive overload |
| **Idle/Reflection** | ⚪ **Special State** | Triggered when pause exceeds 5-second "frontier"; movement to reflection state | User has stopped to think or left workstation |

---

## Output Files

The pipeline generates several key output files:

### **Data Processing Outputs**

| File | Format | Contents | Size (typical) |
|------|--------|----------|----------------|
| `merged_first_sections.csv` | CSV | All keystroke events merged; one row per keystroke press/release | 500MB-2GB* |
| `session_cogload_metrics.csv` | CSV | Session-level features; one row per participant-session | 1-10MB |

*Depends on data volume (16M+ events typical)

### **Model Outputs**

| File | Format | Purpose |
|------|--------|---------|
| `rf_cogload_model_complex.pkl` | Pickle | Trained Random Forest model (200 trees); ready for inference |
| `feature_scaler_complex.pkl` | Pickle | Feature scaling parameters; must be applied before prediction |

**Note**: Both pickle files are required for the real-time testing module.

---

## Usage Guide

### **Quick Start (Complete Pipeline)**

Run all stages sequentially:

```bash
# 1. Merge raw data
python Data-merging.py

# 2. Validate data integrity
python data_understanding.py

# 3. Engineer features
python Data_processing.py

# 4. Train model
python Model_training.py

# 5. Launch real-time monitoring
python testing_model.py
```

### **Using Existing Model (Skip Training)**

If you already have trained model files:

```bash
# Skip stages 1-4, go directly to real-time testing
python testing_model.py
```

### **Custom Data Processing**

Modify input paths in each script to process custom keystroke datasets:

```python
# In Data-merging.py
INPUT_DIR = r"c:\path\to\your\keystroke\files"

# In data_understanding.py
INPUT_FILE = r"c:\path\to\your\raw_data.csv"
```

---

## Troubleshooting

### **Common Issues & Solutions**

| Issue | Cause | Solution |
|-------|-------|----------|
| **ImportError: No module named 'sklearn'** | Missing dependencies | Run `pip install -r requirements.txt` |
| **FileNotFoundError: merged_first_sections.csv** | Earlier pipeline stage not completed | Run `Data-merging.py` first |
| **BACKSPACE_RATIO is always zero** | Backspace encoding not recognized | Check data in `data_understanding.py` and adjust encoding detection |
| **Model performance is poor (R² < 0.7)** | Insufficient training data or poor feature quality | Verify data quality in `data_understanding.py`; collect more samples |
| **Real-time GUI not updating** | Keystroke monitoring permission issue | Grant Python/tkinter keyboard monitoring permissions in OS settings |
| **Memory error during merge** | Dataset too large for RAM | Process data in smaller batches or increase available RAM |

### **Debugging Tips**

- **Enable verbose logging**: Add `print()` statements in each script to monitor processing stages
- **Validate intermediate outputs**: Inspect CSV files with `pandas.read_csv()` to verify data correctness
- **Check model weights**: Use `pickle.load()` to inspect trained model structure before deployment
- **Monitor feature scaling**: Verify StandardScaler fitting on training data matches test data distribution

---

## Technical Details

### **Feature Engineering Mathematics**

**Inter-Keystroke Interval (IKI)**:
$$\text{IKI}_i = \text{PRESS\_TIME}_{i+1} - \text{RELEASE\_TIME}_i$$

**IKI Coefficient of Variation (IKI_CV)**:
$$\text{IKI\_CV} = \frac{\sigma(\text{IKI})}{\mu(\text{IKI})} \times 100\%$$
(Higher CV indicates rhythm inconsistency, suggesting stress)

**Pause Ratio**:
$$\text{PAUSE\_RATIO} = \frac{\sum \text{IKI} > \text{threshold}}{\text{total\_session\_duration}}$$

### **Random Forest Configuration**

```python
RandomForestRegressor(
    n_estimators=200,        # 200 trees in ensemble
    max_depth=15,            # Tree depth limit
    min_samples_split=5,     # Minimum samples to split node
    random_state=42          # Reproducibility
)
```

### **Data Normalization**

All features are normalized using StandardScaler before model training:

$$x_{\text{scaled}} = \frac{x - \mu}{\sigma}$$

Where μ is the mean and σ is the standard deviation of each feature.

---

## Notes

- **Data Privacy**: Ensure all keystroke data is handled according to institutional privacy policies and user consent
- **Validation**: Always review outputs at each stage for data quality before proceeding
- **Model Retraining**: Periodically retrain the model with new data to maintain accuracy
- **System Requirements**: Real-time monitoring requires system keyboard access; may need OS-level permissions
- **Performance**: Processing 16M+ keystroke events may take 1-2 hours depending on hardware

---

## Contact & Support

For questions or issues with the pipeline, refer to inline code comments or review the data understanding stage output for data-specific issues.