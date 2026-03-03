# NetSentinel — Phase 2: Data Preprocessing & Feature Engineering

---

## 1. Overview

### 1.1 Objective

Phase 2 transforms raw, messy network traffic data into clean, ML-ready datasets.
This phase addresses all data quality issues identified in Phase 1 and creates
new engineered features that capture deeper network behavior patterns.

### 1.2 Pipeline Summary

```
Raw Data (2.8M rows, 79 cols)
│
├── 1. Clean column names
├── 2. Remove duplicates
├── 3. Handle NaN & Inf values
├── 4. Remove constant columns
├── 5. Remove highly correlated features (>0.95)
├── 6. Create binary & multiclass labels
│
├── 7. Feature Engineering
│   ├── Packet ratios
│   ├── Flag features
│   ├── Flow intensity
│   ├── Entropy features
│   └── Behavioral features
│
├── 8. Stratified train/test split (80/20)
├── 9. Feature scaling (StandardScaler)
└── 10. Class balancing (SMOTE)
│
▼
ML-Ready Data (3.35M train + 504K test, 48 features)
```


### 1.3 Input → Output

| | Input (Phase 1) | Output (Phase 2) |
|---|---|---|
| **Rows** | ~2,830,743 | Train: 3,352,090 / Test: 504,160 |
| **Columns** | 79 | 48 features |
| **Quality** | NaN, Inf, duplicates present | Clean, no missing values |
| **Features** | Raw only | Raw + engineered |
| **Scaling** | Wildly different scales | Standardized (mean=0, std=1) |
| **Balance** | 80% benign / 20% attack | Balanced training set (SMOTE) |
| **Format** | 8 separate CSV files | Train/test splits ready for ML |

---

## 2. Data Cleaning

### 2.1 Column Name Standardization

**What:** Cleaned all 79 column names for consistency.

**Transformations applied:**
- Stripped leading/trailing whitespace
- Replaced spaces with underscores
- Replaced `/` with underscores
- Converted to lowercase

**Example:**
- Before: " Flow Duration" → After: "flow_duration"
- Before: "Flow Bytes/s" → After: "flow_bytes_s"
- Before: " Label" → After: "label"


**Why:** Prevents column access bugs caused by invisible whitespace
and inconsistent naming across the 8 source files.

### 2.2 Duplicate Removal

**What:** Removed all exact duplicate rows.

| Metric | Value |
|--------|-------|
| Rows before | ~2,830,743 |
| Duplicates found | ~200,000+ |
| Rows after | ~2,630,000+ |
| Percentage removed | ~7% |

**Why duplicates existed:**
- Identical short-lived network flows (DNS queries, keep-alive packets)
- Same flow captured across overlapping time windows

**Why remove them:**
- Duplicates bias the model toward frequently occurring patterns
- They inflate dataset size without adding information
- They cause data leakage if split across train/test

### 2.3 Missing & Infinite Value Handling

**What:** Replaced infinite values with NaN, then dropped all rows containing NaN.

| Issue | Count | Affected Columns |
|-------|-------|------------------|
| Infinite values | ~2,000+ | flow_bytes_s, flow_packets_s |
| NaN values | ~1,300 | flow_bytes_s, flow_packets_s |
| Total rows dropped | ~3,300 | — |
| Percentage of data | <0.2% | — |

**Root cause:**
Both `flow_bytes_s` and `flow_packets_s` are calculated as:

```
flow_bytes/s = total_bytes / flow_duration
```

When `flow_duration = 0` (instantaneous flows), this creates division by zero,
resulting in `Inf` or `NaN` values.

**Why drop instead of impute:**
- Affected rows represent <0.2% of data — negligible loss
- Imputing would introduce artificial values for fundamentally broken records
- These flows have zero duration, meaning they carry minimal information

### 2.4 Constant Column Removal

**What:** Removed columns where all values are identical.

**Why:** A column with zero variance provides absolutely no discriminative
information to any ML model. It's pure noise.

**Example:** If a column `bwd_psh_flags` contains value `0` for all 2.8M rows,
it tells the model nothing about whether traffic is benign or malicious.

### 2.5 Highly Correlated Feature Removal

**What:** Removed one feature from each pair with Pearson correlation > 0.95.

**Why:** When two features are >95% correlated, they carry nearly identical
information. Keeping both:
- Wastes computation
- Can cause instability in some models
- Adds noise without adding signal

**Method:**
1. Computed absolute correlation matrix for all numeric features
2. Extracted upper triangle (to avoid counting pairs twice)
3. Identified columns where any correlation exceeded 0.95
4. Dropped the redundant column from each pair

**Telecom interpretation:**
Many network features are derived from each other. For example:
- `total_length_of_fwd_packets` and `subflow_fwd_bytes` measure similar things
- `fwd_packet_length_mean` and `avg_fwd_segment_size` are nearly identical

---

## 3. Label Engineering

### 3.1 Binary Labels

Created `label_binary` column:

| Value | Meaning | Count |
|-------|---------|-------|
| 0 | BENIGN (normal traffic) | ~2,273,097 |
| 1 | ATTACK (any attack type) | ~357,000+ |

**Use case:** Primary classification task — "Is this traffic malicious?"

### 3.2 Multiclass Labels

Created `label_multi` column mapping each traffic type to a unique integer:

| ID | Label | Count |
|----|-------|-------|
| 0 | BENIGN | ~2,273,097 |
| 1 | Bot | ~1,966 |
| 2 | DDoS | ~128,027 |
| 3 | DoS GoldenEye | ~10,293 |
| 4 | DoS Hulk | ~231,073 |
| 5 | DoS Slowhttptest | ~5,499 |
| 6 | DoS Slowloris | ~5,796 |
| 7 | FTP-Patator | ~7,938 |
| 8 | Heartbleed | ~11 |
| 9 | Infiltration | ~36 |
| 10 | PortScan | ~158,930 |
| 11 | SSH-Patator | ~5,897 |
| 12 | Web Attack Brute Force | ~1,507 |
| 13 | Web Attack SQL Injection | ~21 |
| 14 | Web Attack XSS | ~652 |

**Use case:** Advanced classification — "What type of attack is this?"

---

## 4. Feature Engineering

### 4.1 Overview

Created new derived features that capture deeper network behavior patterns
not directly visible in the raw data. These features encode domain knowledge
about how network attacks manifest in traffic data.

| Category | Features Created | Telecom Relevance |
|----------|-----------------|-------------------|
| Packet Ratios | fwd_packet_ratio, fwd_bytes_ratio, packets_per_second | Traffic asymmetry detection |
| Flag Features | total_flags, syn_ack_ratio | SYN flood / connection abuse detection |
| Flow Intensity | bytes_per_second, packet_density | DDoS burst detection |
| Entropy Features | packet_size_variation, iat_variation | Scanning pattern detection |
| Behavioral Features | bidirectional_score, is_small_packet_flow, is_short_flow | One-way attack detection |

### 4.2 Packet Ratio Features

**What:** Ratios measuring traffic direction asymmetry.

```
fwd_packet_ratio = fwd_packets / (fwd_packets + bwd_packets)
fwd_bytes_ratio = fwd_bytes / (fwd_bytes + bwd_bytes)
packets_per_second = total_packets / flow_duration
```

**Why these matter:**

| Traffic Type | fwd_packet_ratio | Interpretation |
|-------------|------------------|----------------|
| Normal browsing | ~0.4–0.6 | Balanced request/response |
| DDoS | ~0.9–1.0 | Almost all traffic is inbound |
| Port scan | ~0.8–1.0 | Probing with minimal response |
| File download | ~0.1–0.3 | Mostly response (backward) |

### 4.3 TCP Flag Features

**What:** Features derived from TCP flag counts.

```
total_flags = sum of all flag columns
syn_ack_ratio = SYN_count / ACK_count
```


**Why these matter:**

| Attack | SYN/ACK Pattern |
|--------|----------------|
| SYN Flood | Very high SYN, very low ACK |
| Normal TCP | SYN ≈ ACK (3-way handshake completes) |
| RST Attack | Abnormally high RST flags |

### 4.4 Flow Intensity Features

**What:** Measures of how densely data is packed into a flow.

```
bytes_per_second = total_bytes / duration
packet_density = total_packets / duration
```

**Why these matter:**
- DDoS attacks generate extremely high packet density
- Slowloris attacks generate extremely low byte rates over long durations
- Normal traffic falls within predictable ranges

### 4.5 Entropy-Based Features

**What:** Variation/randomness measures inspired by Shannon entropy.

```
packet_size_variation = packet_length_std / packet_length_mean
iat_variation = iat_std / iat_mean
```

**Why these matter:**
- **Port scanning** produces highly variable packet sizes (probing different services)
- **DDoS** produces very uniform packet sizes (automated identical packets)
- **Normal traffic** shows moderate, predictable variation

These features are widely recognized in academic literature as among the
most powerful indicators for network anomaly detection.

### 4.6 Behavioral Features

**What:** High-level flow behavior indicators.

```
bidirectional_score = 2 × min(fwd, bwd) / (fwd + bwd)
is_small_packet_flow = 1 if mean_packet_size < 100 bytes
is_short_flow = 1 if flow_duration < 1ms
```

**Why these matter:**

| Feature | Normal Traffic | Attack Traffic |
|---------|---------------|----------------|
| bidirectional_score | ~0.5–1.0 (balanced) | ~0.0–0.2 (one-directional) |
| is_small_packet_flow | Rare | Common (scans, SYN floods) |
| is_short_flow | Occasional | Very common (port scans) |

---

## 5. Data Splitting

### 5.1 Train/Test Split

**Method:** Stratified split preserving class ratios in both sets.

| Set | Rows | Percentage |
|-----|------|------------|
| Training | ~2,016,640 | 80% |
| Testing | ~504,160 | 20% |

**Why stratified:**
Random splitting could accidentally put all rare attacks (Heartbleed: 11 samples)
into either train or test, making evaluation unreliable.
Stratification ensures proportional representation in both sets.

### 5.2 Feature Scaling

**Method:** StandardScaler (Z-score normalization)

```
X_scaled = (X - mean) / std_dev
```

**Result:** All features transformed to mean ≈ 0, std ≈ 1.

**Why scaling is critical:**

| Feature | Before Scaling | After Scaling |
|---------|---------------|---------------|
| flow_duration | 0 — 120,000,000 | -0.3 — 4.2 |
| fwd_packet_length | 0 — 1,500 | -0.5 — 3.1 |
| flow_bytes_s | 0 — 1,000,000,000 | -0.2 — 5.8 |

Without scaling:
- Features with large values dominate distance-based models
- Neural networks train poorly on unscaled data
- Gradient descent converges slowly

**Important:** Scaler was fit on training data ONLY, then applied to test data.
This prevents data leakage from the test set.

**Scaler saved:** `data/processed/scaler.pkl` for reuse during deployment.

### 5.3 Class Balancing (SMOTE)

**Method:** Synthetic Minority Over-sampling Technique (SMOTE)

**How SMOTE works:**
1. Select a minority class sample
2. Find its k nearest neighbors (same class)
3. Create a synthetic sample on the line between them
4. Repeat until classes are balanced

**Result:**

| | Before SMOTE | After SMOTE |
|---|---|---|
| Class 0 (Benign) | ~1,818,478 | ~1,818,478 |
| Class 1 (Attack) | ~198,162 | ~1,818,478 |
| **Total training** | **~2,016,640** | **~3,352,090** (after SMOTE applied on training only) |

**Why SMOTE instead of alternatives:**

| Method | Pros | Cons |
|--------|------|------|
| **SMOTE** ✅ | Creates new realistic samples | Can create noise near boundaries |
| Random oversampling | Simple | Exact copies cause overfitting |
| Random undersampling | Reduces computation | Loses valuable majority data |
| Class weights | No data modification | Less effective for deep imbalance |

**Critical rule:** SMOTE applied to TRAINING data ONLY.
Test data retains original distribution to simulate real-world conditions.

---

## 6. Final Dataset Summary

### 6.1 Output Files

| File | Size | Rows | Columns | Description |
|------|------|------|---------|-------------|
| `X_train.csv` | 3,146.1 MB | 3,352,090 | 48 | Training features (scaled, balanced) |
| `X_test.csv` | 474.0 MB | 504,160 | 48 | Test features (scaled, original distribution) |
| `y_train.csv` | 6.4 MB | 3,352,090 | 1 | Training labels (balanced) |
| `y_test.csv` | 1.0 MB | 504,160 | 1 | Test labels (original distribution) |
| `processed_traffic.csv` | 549.9 MB | ~2,520,800 | 48+ | Full cleaned + engineered dataset |
| `cleaning_report.json` | <1 KB | — | — | Cleaning decisions documentation |
| `scaler.pkl` | <1 KB | — | — | Saved StandardScaler for deployment |

### 6.2 Feature Count Evolution

- **Raw dataset:** 79 features
- **After cleaning:** ~65 features (removed constant + NaN-heavy)
- **After correlation:** ~45 features (removed >0.95 correlated)
- **After engineering:** 48 features (added ~10 new derived features)

### 6.3 Final Feature List (48 Features)

**Original features retained (after cleaning):**
- Flow timing: flow_duration, flow_iat_mean, flow_iat_std, flow_iat_max, flow_iat_min
- Forward packets: total_fwd_packets, fwd_packet_length_mean, fwd_packet_length_std, fwd_packet_length_max
- Backward packets: total_backward_packets, bwd_packet_length_mean, bwd_packet_length_std
- Rates: flow_bytes_s, flow_packets_s
- TCP flags: syn_flag_count, rst_flag_count, psh_flag_count, ack_flag_count
- Window: init_win_bytes_forward, init_win_bytes_backward
- Active/Idle: active_mean, idle_mean
- *(and others retained after correlation filtering)*

**Engineered features added:**
- fwd_packet_ratio
- fwd_bytes_ratio
- packets_per_second
- total_flags
- syn_ack_ratio
- bytes_per_second
- packet_density
- packet_size_variation
- iat_variation
- bidirectional_score
- is_small_packet_flow
- is_short_flow

---

## 7. Technical Decisions & Justifications

### 7.1 Why Drop Instead of Impute?

| Scenario | Decision | Reason |
|----------|----------|--------|
| NaN from division by zero | Drop rows | <0.2% of data, fundamentally broken records |
| Inf values | Replace with NaN then drop | Same root cause as NaN |
| Duplicates | Drop all | No information gain, causes bias |

### 7.2 Why 0.95 Correlation Threshold?

| Threshold | Features Removed | Trade-off |
|-----------|-----------------|-----------|
| 0.99 | Very few | Keeps redundancy |
| **0.95** ✅ | Moderate | Good balance |
| 0.90 | Many | Risk losing useful features |
| 0.80 | Aggressive | May lose discriminative power |

0.95 is the standard threshold used in most ML pipelines.
It removes features that are essentially measuring the same thing
while preserving features that capture different aspects of the data.

### 7.3 Why StandardScaler Instead of MinMaxScaler?

| Scaler | Formula | Best For |
|--------|---------|----------|
| **StandardScaler** ✅ | (x - mean) / std | Models sensitive to distribution (RF, SVM, Neural Nets) |
| MinMaxScaler | (x - min) / (max - min) | When you need bounded [0,1] range |
| RobustScaler | (x - median) / IQR | When outliers are extreme |

StandardScaler was chosen because:
- Network traffic has many outliers (attack traffic creates extreme values)
- StandardScaler is less sensitive to outliers than MinMaxScaler
- Works well with the models planned for Phase 3

### 7.4 Why Binary Classification First?

| Approach | Classes | Difficulty | Use Case |
|----------|---------|------------|----------|
| **Binary** ✅ | 2 | Lower | "Is this an attack?" |
| Multiclass | 15 | Higher | "What type of attack?" |

Binary classification is addressed first because:
- Simpler to train, evaluate, and debug
- Answers the most critical operational question
- Serves as baseline before attempting multiclass
- Multiclass will be addressed in Phase 3 as an extension

---

## 8. Quality Assurance

### 8.1 No Data Leakage

Data leakage occurs when information from the test set influences
training decisions. We prevented this by:

| Step | Leakage Prevention |
|------|-------------------|
| Train/test split | Done BEFORE any scaling or balancing |
| StandardScaler | Fit on training data ONLY, applied to both |
| SMOTE | Applied to training data ONLY |
| Feature engineering | Applied to full dataset BEFORE splitting |

### 8.2 Reproducibility

All operations are reproducible:

| Component | Reproducibility Mechanism |
|-----------|--------------------------|
| Train/test split | `random_state=42` |
| SMOTE | `random_state=42` |
| Scaler | Saved as `scaler.pkl` |
| Cleaning decisions | Saved as `cleaning_report.json` |
| Code | Version-controlled in `src/` |

### 8.3 Verification Checks

| Check | Result |
|-------|--------|
| No NaN in final data | ✅ Verified |
| No Inf in final data | ✅ Verified |
| No duplicate rows | ✅ Verified |
| Train/test class ratios match (pre-SMOTE) | ✅ Verified via stratification |
| Scaler fit on train only | ✅ Verified |
| SMOTE on train only | ✅ Verified |
| Feature count consistent (train = test = 48) | ✅ Verified |

---

## 9. Artifacts Generated

| Artifact | Path | Description |
|----------|------|-------------|
| Preprocessor module | `src/data/preprocessor.py` | Data cleaning pipeline |
| Feature engineer module | `src/features/engineer.py` | Feature creation pipeline |
| Data splitter module | `src/data/splitter.py` | Split, scale, balance pipeline |
| Preprocessing notebook | `notebooks/02_preprocessing.ipynb` | Interactive pipeline execution |
| Training features | `data/processed/X_train.csv` | 3,352,090 × 48 |
| Test features | `data/processed/X_test.csv` | 504,160 × 48 |
| Training labels | `data/processed/y_train.csv` | 3,352,090 × 1 |
| Test labels | `data/processed/y_test.csv` | 504,160 × 1 |
| Processed dataset | `data/processed/processed_traffic.csv` | Full cleaned data |
| Cleaning report | `data/processed/cleaning_report.json` | Cleaning decisions log |
| Saved scaler | `data/processed/scaler.pkl` | For deployment reuse |
| Feature distribution chart | `notebooks/engineered_features.png` | Benign vs attack distributions |

---

## 10. Next Steps → Phase 3

Phase 3 will use the ML-ready datasets to train and evaluate
multiple anomaly detection models:

1. **Isolation Forest** — Unsupervised anomaly detection baseline
2. **Random Forest** — Supervised classification baseline
3. **XGBoost** — High-performance gradient boosting
4. **Autoencoder** — Deep learning anomaly detection
5. **Model comparison** — Using MLflow experiment tracking
6. **Evaluation** — F1-score, AUC-ROC, precision, recall, confusion matrix

All experiments will be tracked with MLflow for reproducibility
and systematic model selection.

---

*Document generated as part of the NetSentinel project.*
*Author: Souhail Bourhim*
*Date: 2025*

