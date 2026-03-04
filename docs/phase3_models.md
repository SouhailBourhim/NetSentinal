# NetSentinel — Phase 3: Model Training, Evaluation & Analysis

---

## 1. Overview

### 1.1 Objective
Phase 3 trains, evaluates, and compares multiple anomaly detection models on the preprocessed CIC-IDS2017 dataset. This phase goes beyond standard evaluation by investigating data leakage, performing temporal validation, and testing hybrid detection strategies.

### 1.2 Models Trained

| Model | Type | Approach |
|---|---|---|
| **Isolation Forest** | Unsupervised | Isolates outliers via random partitioning |
| **Random Forest** | Supervised | Ensemble of decision trees with majority voting |
| **XGBoost** | Supervised | Sequential gradient boosting |
| **Autoencoder** | Deep Learning | Reconstruction error-based anomaly detection |

### 1.3 Key Finding
Initial evaluation showed AUC = 1.0 for tree-based models. Investigation revealed this was inflated by near-duplicate data leakage. Temporal validation provided realistic performance estimates, and a hybrid supervised + unsupervised approach was developed to improve detection of unknown attack types.

---

## 2. Standard Evaluation Results

### 2.1 Initial Results (Random Split with SMOTE)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| XGBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Random Forest | 0.9999 | 0.9999 | 0.9999 | 0.9999 | 1.0000 |
| Isolation Forest | — | — | — | — | — |
| Autoencoder | — | — | — | — | — |

### 2.2 Why These Results Are Suspicious
An AUC of 1.0 indicates either a perfect model or a data issue. Further investigation was warranted to determine the root cause.

---

## 3. Data Leakage Investigation

### 3.1 Near-Duplicate Analysis
A nearest-neighbor analysis on 10,000 test samples revealed:

| Metric | Value |
|---|---|
| Exact matches (distance = 0) | 107 (1.1%) |
| Near matches (distance < 0.01) | 7,316 (73.2%) |
| Near matches (distance < 0.1) | 9,011 (90.1%) |

> **Note:** 73.2% of test samples had near-identical counterparts in the training set.

### 3.2 Root Cause
CIC-IDS2017 generates thousands of nearly identical network flows during each attack session. Random train/test splitting places copies of the same attack session in both sets, allowing models to "memorize" rather than "generalize."

### 3.3 Verification Strategy
Three alternative evaluation methods were tested to obtain realistic metrics.

---

## 4. Robust Evaluation Results

### 4.1 Evaluation Strategies

| Strategy | Description | Leakage Risk |
|---|---|---|
| **Standard random split** | 80/20 with SMOTE | 🔴 High |
| **Deduplicated split** | Remove near-duplicates before splitting | 🟡 Low |
| **5-Fold cross-validation** | Stratified CV on deduplicated data | 🟡 Low |
| **Temporal split** | Train Mon-Wed, Test Thu-Fri | 🟢 None |

### 4.2 Results Comparison

| Strategy | AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| Standard (leaked) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Deduplicated | 1.0000 | 0.9978 | 0.9969 | 0.9986 |
| Cross-validation | 1.0000 ± 0.0000 | 0.9959 ± 0.0004 | 0.9951 | 0.9967 |
| **Temporal split** | **0.8015** | **0.4409** | **0.9977** | **0.2830** |

### 4.3 Key Observations
1. **Deduplicated and CV results remain near-perfect** because CIC-IDS2017 attack signatures are inherently distinct from normal traffic within the same time period.
2. **Temporal split reveals the real challenge:** the model cannot detect attack types it has never seen before.
3. **Precision remains excellent (99.77%):** when the model flags traffic as an attack, it is almost always correct.
4. **Recall is low (28.30%):** the model misses most attacks from previously unseen categories.

---

## 5. Detection by Attack Type (Temporal Split)

### 5.1 Results

| Attack Type | Total | Detected | Rate | Explanation |
|---|---|---|---|---|
| ✅ Web Attack Brute Force | 1,507 | 1,433 | 95.1% | Similar to FTP/SSH Patator in training |
| ✅ Web Attack XSS | 652 | 626 | 96.0% | Repetitive patterns generalize well |
| ✅ Web Attack SQL Injection | 21 | 18 | 85.7% | Similar behavior to brute force |
| 🟡 DDoS | 128,027 | 86,800 | 67.8% | Partially similar to DoS in training |
| 🔴 PortScan | 158,930 | 888 | 0.6% | Fundamentally different pattern |
| 🔴 Bot | 1,966 | 0 | 0.0% | Designed to mimic normal traffic |
| 🔴 Infiltration | 36 | 0 | 0.0% | Too subtle, too few samples |
| 🟢 Benign (correct) | 871,074 | 870,555 | 99.9% | Excellent — very few false alarms |

### 5.2 Interpretation
Generalization depends heavily on behavioral similarity:

| Training Attack | Test Attack | Detection Success |
|---|---|---|
| **FTP/SSH Patator** (repeated login) | **Web Brute Force** (repeated login) | ✅ HIGH (same behavior) |
| **DoS Hulk/Slowloris** (flood traffic) | **DDoS** (distributed flood) | 🟡 PARTIAL (similar but different scale) |
| **No equivalent** (port probing) | **PortScan** (never seen this pattern) | 🔴 NONE |
| **No equivalent** (stealthy, distributed) | **Bot** (designed to evade) | 🔴 NONE |

---

## 6. Hybrid Model

### 6.1 Rationale
To address the limitation of supervised-only detection, a hybrid approach was developed combining:
* **XGBoost (supervised):** Detects known attack signatures.
* **Isolation Forest (unsupervised):** Detects statistical anomalies.

This mirrors real-world telecom security architecture where signature-based IDS and anomaly-based IDS work together.

### 6.2 Combination Strategies Tested

| Strategy | F1 | Precision | Recall | AUC |
|---|---|---|---|---|
| XGBoost Only (t=0.5) | 0.4318 | 0.9972 | 0.2756 | 0.7436 |
| Isolation Forest Only | 0.1096 | 0.6517 | 0.0598 | 0.7900 |
| **Weighted 70% XGB + 30% IF** | **0.4429** | **0.9964** | **0.2847** | **0.8033** |
| Weighted 60/40 | 0.4414 | 0.9961 | 0.2835 | 0.8023 |
| Weighted 50/50 | 0.4363 | 0.9868 | 0.2801 | 0.8015 |
| OR Logic (XGB≥0.3, IF≥0.5) | 0.4314 | 0.8960 | 0.2840 | 0.7989 |
| Max Score | 0.4246 | 0.8775 | 0.2800 | 0.7989 |

### 6.3 Best Strategy: Weighted 70% XGB + 30% IF

| Metric | Value |
|---|---|
| **F1-Score** | 0.4429 |
| **Precision** | 0.9964 |
| **Recall** | 0.2847 |
| **AUC-ROC** | 0.8033 |
| **False alarms** | 300 / 871,074 (0.03%) |

### 6.4 Hybrid Impact on Detection

| Attack Type | XGBoost Only | Hybrid | Improvement |
|---|---|---|---|
| DDoS | 60.8% | 62.8% | +2.0% |
| Web Brute Force | 87.1% | 88.1% | +1.0% |
| Web XSS | 91.0% | 91.6% | +0.6% |
| Web SQL Injection | 71.4% | 76.2% | +4.8% |
| PortScan | 0.3% | 0.3% | +0.0% |
| Bot | 0.0% | 0.0% | +0.0% |

### 6.5 Honest Assessment
The hybrid approach provides marginal improvement (+8% AUC, +3% recall) but does NOT solve the fundamental generalization problem. Bot and PortScan remain undetectable because:
1. **Bot traffic** is specifically designed to mimic normal behavior.
2. **PortScan** has a fundamentally different pattern from any training attack.
3. The **Isolation Forest contamination** is not sensitive enough to distinguish these subtle anomalies from normal traffic variation.

---

## 7. Real-World Implications

### 7.1 What This Means for Telecom Deployment

| Aspect | Implication |
|---|---|
| **False alarm rate** | Excellent (0.03%) — operationally viable for NOC teams |
| **Known attack detection** | Strong — effectively detects DoS, brute force, web attacks |
| **Unknown attack detection** | Weak — requires continuous retraining with new attack data |
| **Deployment model** | Hybrid supervised + unsupervised is the industry standard |
| **Continuous learning** | Model must be regularly updated with new attack signatures |

### 7.2 Recommendations for Production
1. **Regular retraining** with new labeled attack data (weekly/monthly).
2. **Threshold tuning** based on SOC team capacity and risk tolerance.
3. **Multi-layer defense:** deploy this model as one layer alongside rule-based IDS.
4. **Feature monitoring:** detect drift in traffic patterns that indicate model degradation.
5. **Human-in-the-loop:** flagged anomalies should be reviewed by analysts.

---

## 8. Artifacts Generated

| Artifact | Path | Description |
|---|---|---|
| Base model class | `src/models/base_model.py` | Abstract detector interface |
| Isolation Forest | `src/models/isolation_forest.py` | Unsupervised detector |
| Random Forest | `src/models/random_forest.py` | Supervised detector |
| XGBoost | `src/models/xgboost_model.py` | Gradient boosting detector |
| Autoencoder | `src/models/autoencoder.py` | Deep learning detector |
| Model comparator | `src/models/comparator.py` | Comparison utilities |
| Robust evaluator | `src/models/robust_evaluator.py` | Leakage detection |
| Model config | `configs/model_config.yaml` | Hyperparameters |
| Training notebook | `notebooks/03_model_training.ipynb` | Standard training |
| Leakage analysis | `notebooks/04_leakage_check.ipynb` | Leakage investigation |
| Robust evaluation | `notebooks/05_robust_evaluation.ipynb` | Multi-strategy evaluation |
| Improved temporal | `notebooks/06_improved_temporal.ipynb` | Temporal optimization |
| Hybrid model | `notebooks/07_hybrid_model.ipynb` | Combined approach |
| Model comparison chart | `notebooks/model_comparison.png` | Visual comparison |
| ROC comparison | `notebooks/roc_comparison.png` | ROC overlay |
| Temporal evaluation | `notebooks/temporal_evaluation.png` | Temporal results |
| Hybrid analysis | `notebooks/hybrid_analysis.png` | Hybrid visualization |

---

## 9. Next Steps → Phase 4
Phase 4 will deploy the best model as a production-ready service:
1. **FastAPI** — REST API for real-time anomaly prediction.
2. **Streamlit** — Monitoring dashboard for network security teams.
3. **Docker** — Containerized deployment.
4. **Complete README** — Project documentation for GitHub.

---

*Document generated as part of the NetSentinel project.*
*Author: Souhail Bourhim*
*Date: 2025*