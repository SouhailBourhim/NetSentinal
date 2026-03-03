# NetSentinel — Phase 1: Data Acquisition & Exploration

---

## 1. Overview

### 1.1 Objective

The goal of Phase 1 is to acquire, load, and thoroughly explore the network
traffic dataset before any preprocessing or model building. This phase answers
three fundamental questions:

1. **What data do we have?** — Size, structure, features
2. **What does it look like?** — Distributions, patterns, anomalies
3. **What problems exist?** — Missing values, duplicates, imbalance

### 1.2 Dataset

| Property | Value |
|----------|-------|
| **Name** | CIC-IDS2017 |
| **Source** | Canadian Institute for Cybersecurity, University of New Brunswick |
| **URL** | https://www.unb.ca/cic/datasets/ids-2017.html |
| **Format** | CSV (MachineLearningCSV.zip) |
| **Capture Period** | Monday July 3 — Friday July 7, 2017 |
| **Environment** | Simulated enterprise network with realistic traffic |

### 1.3 Why This Dataset?

CIC-IDS2017 was chosen for the following reasons:

- **Labeled data** — Every flow is tagged as BENIGN or a specific attack type
- **Realistic traffic** — Generated in a controlled but realistic network environment
- **Modern attacks** — Includes DDoS, port scans, brute force, web attacks, botnets
- **Rich features** — 79 network flow features extracted using CICFlowMeter
- **Well-documented** — Extensively used in academic research
- **Large scale** — ~2.8 million records, suitable for ML training

---

## 2. Data Acquisition

### 2.1 Files Downloaded

The dataset consists of 8 CSV files, one per capture session:

| File | Day | Content |
|------|-----|---------|
| `Monday-WorkingHours.pcap_ISCX.csv` | Monday | Benign traffic only (baseline) |
| `Tuesday-WorkingHours.pcap_ISCX.csv` | Tuesday | FTP-Patator, SSH-Patator |
| `Wednesday-WorkingHours.pcap_ISCX.csv` | Wednesday | DoS Slowloris, DoS Slowhttptest, DoS Hulk, DoS GoldenEye, Heartbleed |
| `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | Thursday AM | Web Attack Brute Force, XSS, SQL Injection |
| `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` | Thursday PM | Infiltration |
| `Friday-WorkingHours-Morning.pcap_ISCX.csv` | Friday AM | Bot |
| `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | Friday PM | DDoS |
| `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | Friday PM | PortScan |

### 2.2 Storage

```
NetSentinel/
└── data/
    └── raw/
        ├── Monday-WorkingHours.pcap_ISCX.csv
        ├── Tuesday-WorkingHours.pcap_ISCX.csv
        ├── Wednesday-WorkingHours.pcap_ISCX.csv
        ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
        ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
        ├── Friday-WorkingHours-Morning.pcap_ISCX.csv
        ├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
        └── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
```

---

## 3. Data Loading & Merging

### 3.1 Process

1. All 8 CSV files were loaded individually using `pandas.read_csv()`
2. Column names were stripped of leading/trailing whitespace
3. All DataFrames were concatenated into a single unified DataFrame

### 3.2 Result

| Metric | Value |
|--------|-------|
| **Total rows** | ~2,830,743 |
| **Total columns** | 79 |
| **Memory usage** | ~1.7 GB |

### 3.3 Why Merge?

Working with a single DataFrame:
- Simplifies all downstream operations
- Ensures consistent preprocessing across all days
- Allows cross-day pattern analysis

---

## 4. Data Inspection

### 4.1 Feature Categories

The 79 features can be grouped into the following categories:

| Category | Features | Description |
|----------|----------|-------------|
| **Flow Identification** | Destination Port | Target port of the flow |
| **Flow Timing** | Flow Duration, Flow IAT (Mean/Std/Max/Min) | Duration and inter-arrival times |
| **Packet Counts** | Total Fwd/Bwd Packets, Subflow Fwd/Bwd Packets | Number of packets in each direction |
| **Packet Lengths** | Fwd/Bwd Packet Length (Mean/Std/Max/Min) | Size statistics of packets |
| **Byte Counts** | Total Length of Fwd/Bwd Packets, Flow Bytes/s | Total data volume |
| **Rate Features** | Flow Packets/s, Flow Bytes/s | Speed of data transfer |
| **TCP Flags** | FIN/SYN/RST/PSH/ACK/URG/CWE/ECE Flag Count | TCP connection behavior |
| **Header Info** | Fwd/Bwd Header Length | Protocol overhead |
| **Active/Idle** | Active/Idle (Mean/Std/Max/Min) | Flow activity patterns |
| **Segment Size** | Avg Fwd/Bwd Segment Size | Average data chunk size |
| **Bulk Features** | Fwd/Bwd Avg Bytes/Bulk, Fwd/Bwd Avg Packets/Bulk | Bulk transfer behavior |
| **Subflow** | Subflow Fwd/Bwd Packets, Subflow Fwd/Bwd Bytes | Sub-connection level metrics |
| **Init Win** | Init_Win_bytes_forward/backward | TCP initial window size |
| **Label** | Label | Traffic classification (target variable) |

### 4.2 Data Types

| Type | Count |
|------|-------|
| Float64 | ~55 columns |
| Int64 | ~22 columns |
| Object (string) | 1 column (Label) |

---

## 5. Label Analysis

### 5.1 Traffic Distribution

The dataset contains 15 unique traffic categories:

| Label | Count | Percentage | Type |
|-------|-------|------------|------|
| BENIGN | ~2,273,097 | 80.3% | Normal |
| DoS Hulk | ~231,073 | 8.2% | Attack |
| PortScan | ~158,930 | 5.6% | Attack |
| DDoS | ~128,027 | 4.5% | Attack |
| DoS GoldenEye | ~10,293 | 0.4% | Attack |
| FTP-Patator | ~7,938 | 0.3% | Attack |
| SSH-Patator | ~5,897 | 0.2% | Attack |
| DoS Slowloris | ~5,796 | 0.2% | Attack |
| DoS Slowhttptest | ~5,499 | 0.2% | Attack |
| Bot | ~1,966 | 0.07% | Attack |
| Web Attack Brute Force | ~1,507 | 0.05% | Attack |
| Web Attack XSS | ~652 | 0.02% | Attack |
| Infiltration | ~36 | 0.001% | Attack |
| Web Attack SQL Injection | ~21 | 0.0007% | Attack |
| Heartbleed | ~11 | 0.0004% | Attack |

### 5.2 Binary Distribution

| Category | Count | Percentage |
|----------|-------|------------|
| Benign | ~2,273,097 | 80.3% |
| Attack | ~557,646 | 19.7% |

### 5.3 Key Observations

1. **Severe class imbalance** — 80% benign vs 20% attack
2. **Intra-attack imbalance** — DoS Hulk has 231K samples while Heartbleed has only 11
3. **Implications for modeling:**
   - Accuracy is a misleading metric (predicting all-benign = 80% accuracy)
   - Must use F1-score, precision, recall, and AUC-ROC
   - Must apply class balancing techniques (SMOTE, undersampling)
   - Rare attacks (Heartbleed, SQL Injection) may need special handling

---

## 6. Data Quality Assessment

### 6.1 Missing Values

| Issue | Count | Affected Columns |
|-------|-------|------------------|
| NaN values | ~1,300 | Flow Bytes/s, Flow Packets/s |

**Root cause:** Division by zero when flow duration = 0
**Resolution plan:** Drop affected rows (negligible percentage)

### 6.2 Infinite Values

| Issue | Count |
|-------|-------|
| Infinite values | ~2,000+ |

**Root cause:** Same division-by-zero issue in rate calculations
**Resolution plan:** Replace Inf with NaN, then drop

### 6.3 Duplicate Rows

| Issue | Count |
|-------|-------|
| Exact duplicate rows | ~200,000+ |

**Root cause:** Identical short-lived flows (DNS queries, keep-alive packets)
**Resolution plan:** Remove all exact duplicates

### 6.4 Constant Columns

Some columns may have zero variance (all values identical).
**Resolution plan:** Identify and remove in Phase 2

---

## 7. Feature Correlation Analysis

### 7.1 Methodology

1. Created binary label: `is_attack` (0 = Benign, 1 = Attack)
2. Computed absolute Pearson correlation between each feature and `is_attack`
3. Ranked features by correlation strength

### 7.2 Top 20 Most Correlated Features

| Rank | Feature | Correlation |
|------|---------|-------------|
| 1 | Bwd Packet Length Std | ~0.45 |
| 2 | Subflow Fwd Bytes | ~0.42 |
| 3 | Total Length of Fwd Packets | ~0.41 |
| 4 | Fwd Packet Length Mean | ~0.40 |
| 5 | Average Packet Size | ~0.39 |
| 6 | Flow Bytes/s | ~0.38 |
| 7 | Fwd Packet Length Max | ~0.37 |
| 8 | Avg Fwd Segment Size | ~0.36 |
| 9 | Subflow Fwd Packets | ~0.34 |
| 10 | Total Fwd Packets | ~0.33 |
| 11 | Flow IAT Std | ~0.31 |
| 12 | Flow Duration | ~0.30 |
| 13 | Bwd Packet Length Mean | ~0.29 |
| 14 | Init_Win_bytes_forward | ~0.28 |
| 15 | Fwd IAT Std | ~0.27 |
| 16 | Fwd Header Length | ~0.26 |
| 17 | Bwd Packet Length Max | ~0.25 |
| 18 | Flow IAT Max | ~0.24 |
| 19 | Destination Port | ~0.23 |
| 20 | SYN Flag Count | ~0.22 |

### 7.3 Interpretation

**Packet size features dominate** — attacks tend to generate packets with
different size distributions than normal traffic:
- DDoS: many small identical packets
- Port scans: small probing packets
- Data exfiltration: unusually large packets

**Timing features are significant** — attacks create abnormal timing patterns:
- Port scans: rapid successive connections (low IAT)
- DoS Slowloris: intentionally slow connections (high duration)

**TCP flags correlate with attacks** — SYN floods generate abnormal flag patterns

---

## 8. Key Findings Summary

### 8.1 Dataset Strengths
- ✅ Large scale (2.8M+ records)
- ✅ Well-labeled (15 traffic categories)
- ✅ Rich feature set (79 flow-level features)
- ✅ Realistic attack scenarios

### 8.2 Dataset Challenges
- ⚠️ Severe class imbalance (80/20 benign/attack)
- ⚠️ Intra-attack class imbalance (231K vs 11 samples)
- ⚠️ Data quality issues (NaN, Inf, duplicates)
- ⚠️ Feature scale variance (need normalization)
- ⚠️ Potential feature redundancy (high inter-feature correlation)

### 8.3 Decisions for Phase 2

Based on Phase 1 findings, the following decisions were made:

| Decision | Rationale |
|----------|-----------|
| Drop NaN/Inf rows | Small percentage, safe to remove |
| Remove duplicates | Reduce bias, no information loss |
| Remove constant columns | Zero predictive value |
| Remove highly correlated features (>0.95) | Reduce redundancy |
| Use StandardScaler normalization | Features on vastly different scales |
| Apply SMOTE on training data | Address class imbalance |
| Use F1/AUC as primary metrics | Accuracy is misleading for imbalanced data |
| Start with binary classification | Simpler, then extend to multiclass |
| Focus on top correlated features | Reduce noise, improve model efficiency |

---

## 9. Artifacts Generated

| Artifact | Path | Description |
|----------|------|-------------|
| Exploration notebook | `notebooks/01_exploration.ipynb` | Interactive data exploration |
| Label distribution chart | `notebooks/label_distribution.png` | Visual class distribution |
| Feature correlation chart | `notebooks/feature_correlations.png` | Top 20 correlated features |
| Exploration summary | `notebooks/exploration_summary.json` | Key metrics in JSON format |

---

## 10. Next Steps → Phase 2

Phase 2 will address all identified data quality issues and prepare
ML-ready datasets:

1. **Data Cleaning** — Remove NaN, Inf, duplicates, constant columns
2. **Feature Engineering** — Create new derived features (ratios, entropy, behavioral)
3. **Feature Selection** — Remove highly correlated redundant features
4. **Label Encoding** — Create binary and multiclass label columns
5. **Train/Test Split** — Stratified 80/20 split
6. **Feature Scaling** — StandardScaler normalization
7. **Class Balancing** — SMOTE on training data only

---

*Document generated as part of the NetSentinel project.*
*Author: Souhail Bourhim*
*Date: 2025*