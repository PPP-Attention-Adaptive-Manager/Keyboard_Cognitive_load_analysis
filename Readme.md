# 🧠 Keyboard Cognitive Load Analysis System (V2)

## 📌 Overview

This project is a **real-time behavioral intelligence system** that infers cognitive states (load, stress, attention) from keystroke dynamics.

It combines:

- Deterministic behavioral modeling (feature engineering + proxies)
- Machine learning (RandomForest regression on behavioral space)
- Real-time inference engine (streaming keystroke processing)
- Personalization layer (user-specific adaptation over time)
- Reinforcement learning layer (planned adaptive weighting system)

The goal is to evolve from a static prediction system into a **self-adapting cognitive model per user**.

---

# 🏗️ System Architecture

```
Raw Keyboard Events
        ↓
RealtimeFeatureBuilder
        ↓
Session Feature Extraction
        ↓
Proxy Layer (COGLOAD / STRESS / UNFOCUS)
        ↓
ML Model (RandomForest)
        ↓
Fusion Layer (ML + Static Proxy)
        ↓
User Profile Memory (Personalization)
        ↓
RL Adaptive Layer (future evolution)
        ↓
Final Cognitive State Output
```

---

# ⚙️ Core Modules

## 1. Feature Engineering Layer

Extracts behavioral signals:

- Inter-Key Interval (IKI)
- Hold Time
- Pause Ratio
- Backspace Ratio
- Sentence structure metrics
- Error rate (ML-based or lexical distance)

### Key idea:
> Convert raw keystrokes into **behavioral time-series signals**

---

## 2. Proxy Layer (Static Cognitive Model)

Defines interpretable cognitive states:

### 🧠 Cognitive Load
- typing rhythm instability
- variability in IKI
- pause distribution

### 😰 Stress
- correction frequency
- backspace usage
- error intensity

### 💤 Unfocus
- speed fluctuations
- inconsistent rhythm
- hesitation bursts

---

## 3. ML Layer

Model:
```
RandomForestRegressor
```

Input:
- Selected non-correlated behavioral features

Output:
- Learned approximation of cognitive proxies

Purpose:
> Capture nonlinear relationships that rule-based proxies cannot model

---

## 4. Feature Selection Strategy

To avoid redundancy:

- Greedy correlation filtering
- Mandatory feature injection:
  - BACKSPACE_RATIO
  - MEAN_ERROR_RATE_ML

Goal:
> ensure diversity in behavioral signals

---

## 5. Real-Time Inference Engine

Maintains:

- sliding window of keystrokes
- incremental computation of features
- live update of cognitive state

Key limitation currently:
> fatigue instability due to insufficient temporal smoothing

---

## 6. Fusion Strategy (ML + Static)

Final prediction:

```
Final_State =
    α * ML_prediction +
    (1 - α) * Static_proxy
```

Where:
- α = trust in ML model
- (1 - α) = rule-based robustness

---

# 🧠 Personalization Layer (User Profile)

Each user has a persistent profile:

### Stored statistics:
- baseline IKI
- baseline error rate
- backspace behavior
- pause distribution
- variance patterns

### Purpose:
> Normalize behavior per user instead of global population

---

## Adaptation Mechanism

As user data increases:

```
weight_user_profile ↑
weight_global_model ↓
```

Meaning:

| Usage Time | Behavior |
|------------|----------|
| First session | global model dominant |
| Medium usage | hybrid system |
| Long-term usage | personalized model dominant |

---

# 🔁 Reinforcement Learning Layer (Next Step)

## Objective

Learn optimal weighting between:

- ML prediction
- static proxies
- user profile memory

---

## RL State

```
S = [
  ML_output,
  proxy_output,
  user_deviation,
  temporal_stability
]
```

---

## RL Action

Adjust:

- α (ML weight)
- β (user memory weight)
- γ (proxy weight)

---

## RL Reward

Based on:

- prediction stability
- temporal consistency
- reconstruction error reduction
- user adaptation quality

---

## Final RL Equation

```
Final Prediction =
    α(t) * ML +
    β(t) * UserProfile +
    γ(t) * Proxy
```

---

# 🧍 User Profile System (Memory Layer)

## Structure

```
UserProfile:
    - mean_IKI
    - std_IKI
    - error_rate_baseline
    - stress_sensitivity
    - fatigue_response_curve
```

---

## Learning mechanism

Each session updates:

- exponential moving averages
- variance tracking
- drift detection

---

## Why it matters

Without it:

❌ system is generic  
❌ no adaptation  
❌ unstable long-term predictions  

With it:

✔ personalized cognition model  
✔ stable fatigue estimation  
✔ user-specific behavior mapping  

---

# 📊 Known Issues

## 1. Fatigue instability
Cause:
- insufficient temporal smoothing
- weak long-term memory

Fix:
- rolling normalization
- user baseline anchoring

---

## 2. Static ML model
Current limitation:
- trained once
- no online learning

Fix:
- periodic retraining
- RL-guided updates

---

## 3. Feature drift
Behavior changes over time → model lag

Fix:
- adaptive weighting system
- drift detection module

---

# 🚀 Roadmap

## Phase 1 — Stabilization (current)
- fix real-time feature drift
- improve smoothing
- stabilize fatigue output

---

## Phase 2 — Personalization
- implement UserProfile memory system
- per-user normalization layer
- adaptive baseline correction

---

## Phase 3 — RL Integration
- dynamic weighting system
- reward-based adaptation
- online optimization loop

---

## Phase 4 — Full Cognitive Engine
- self-adaptive per-user system
- continuous learning
- production-ready inference pipeline

---

# 🧪 Running the System

## Training
```bash
python -m V2.training.train_behavior_model
```

## Real-time UI
```bash
python -m V2.ui.dashboard
```

---

# 🧠 Final Vision

This system evolves into:

> A **self-adapting cognitive fingerprint engine**

capable of:

- modeling human typing behavior
- detecting cognitive stress in real time
- adapting to individual users over time
- continuously improving via RL feedback

---

# 🔥 Next Step Recommendation

Implement:

> **RL weighting controller + user profile memory fusion**

This is the step that turns the system from:

```
ML system
→ cognitive system
→ adaptive intelligence system
```