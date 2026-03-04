# NetSentinel — Phase 3: Model Training, Evaluation & Analysis

---

## 1. Objective

Phase 3 focuses on building, training, and evaluating several anomaly detection models on the CIC-IDS2017 dataset, with a specific emphasis on:

- Combining **telecom network knowledge** with **machine learning**
- Understanding the impact of **data leakage and evaluation methodology**
- Evaluating models in **realistic conditions** (temporal split)
- Exploring a **hybrid supervised + unsupervised** detection approach

The goal is not just to reach high scores, but to understand **when and why those scores are trustworthy**.

---

## 2. Models Implemented

### 2.1 Isolation Forest

- Type: **Unsupervised** anomaly detection
- Library: `sklearn.ensemble.IsolationForest`
- Idea:
  - Randomly partitions the feature space
  - Anomalies are isolated in fewer splits (shorter paths in the trees)
  - Designed to detect outliers without labels

### 2.2 Random Forest

- Type: **Supervised** classifier (binary: benign vs attack)
- Library: `sklearn.ensemble.RandomForestClassifier`
- Idea:
  - Ensemble of decision trees trained on bootstrapped samples
  - Each tree votes; final prediction is majority vote
  - Handles high-dimensional, tabular data well
  - Provides feature importance information

### 2.3 XGBoost

- Type: **Supervised** gradient boosting classifier
- Library: `xgboost.XGBClassifier`
- Idea:
  - Trains trees sequentially, each correcting the errors of the previous ones
  - Strong performance on structured/tabular data
  - Handles class imbalance via `scale_pos_weight`
  - Used both with default and tuned hyperparameters

### 2.4 Autoencoder

- Type: **Unsupervised** deep learning anomaly detector
- Library: `TensorFlow / Keras`
- Architecture:
  - Encoder: compresses input features into a small latent vector
  - Decoder: reconstructs the original input from the latent vector
  - Trained **only on benign traffic**
- Detection principle:
  - Normal traffic → low reconstruction error
  - Attack traffic (unseen behavior) → higher reconstruction error
  - If error > threshold → anomaly

### 2.5 Hybrid Model (XGBoost + Isolation Forest)

- Idea:
  - Use **XGBoost** to detect **known attack patterns**
  - Use **Isolation Forest** to detect **statistical anomalies**
  - Combine both scores (e.g., weighted average)
- Motivation:
  - More realistic for telecom security:
    - Signature-like detection for known threats
    - Anomaly-based detection for unknown (zero-day) threats

---

## 3. Datasets and Evaluation Strategies

### 3.1 Base Dataset

- Source: **CIC-IDS2017**
- After Phase 2 preprocessing:
  - Cleaned and engineered features
  - 48 numeric features retained + labels
  - Strong class imbalance handled in Phase 2 (SMOTE for training data)

### 3.2 Standard Random Split (Leaked)

- Train: 80% of rows (stratified)
- Test: 20% of rows
- Class balancing: SMOTE on training data
- Issue: high risk of **data leakage** due to near-duplicate flows across train and test sets.

**Result:** Tree-based models (Random Forest, XGBoost) reached **AUC ≈ 1.0** and F1 ≈ 1.0, but this was later found to be **inflated**.

### 3.3 Data Leakage Investigation

To check for leakage:

1. **Nearest Neighbor Analysis**  
   - For 10,000 random test samples, we computed the distance to their nearest neighbor in the training set.
   - Findings:
     - Exact matches (distance = 0): ~1.1%
     - Near matches (distance < 0.01): ~73.2%
     - Near matches (distance < 0.1): ~90.1%

   → Around **73% of test samples** had near-duplicates in training.

2. **Cross-Validation on Original Data**  
   - 5-fold CV with XGBoost on a sample of the cleaned dataset
   - AUC ≈ 0.99 on average
   - Shows that, even without explicit leakage, CIC-IDS2017 is **inherently easy** for tree-based models due to very distinct attack signatures.

3. **Temporal Split**  
   - Train on early days (**Monday–Wednesday**)
   - Test on later days (**Thursday–Friday**)
   - Eliminates session-level leakage across time.

---

## 4. Robust Evaluation Strategies

After detecting leakage, three robust strategies were used:

### 4.1 Deduplicated Split

- Process:
  - Round all features to 2 decimal places
  - Drop duplicate rows based on rounded feature values
  - Stratified 80/20 train-test split on the deduplicated dataset
- Result:
  - Random Forest & XGBoost still achieved AUC ≈ 1.0 and F1 ≈ 0.996–0.998
- Interpretation:
  - Even after deduplication, attack patterns remain very distinct from benign within the same time period.
  - CIC-IDS2017 is “easy” in this sense.

### 4.2 Cross-Validation (No Temporal Separation)

- Process:
  - 5-fold stratified CV on a deduplicated sample
  - XGBoost as the model
- Result:
  - Mean AUC ≈ 1.0
  - Mean F1 ≈ 0.996 ± small variance
- Interpretation:
  - Confirms the dataset is very separable when train and test come from the *same temporal distribution*.

### 4.3 Temporal Split (Realistic Scenario)

- Train: Monday, Tuesday, Wednesday
- Test: Thursday (Web Attacks, Infiltration) and Friday (Bot, DDoS, PortScan)
- This simulates real deployment:
  > “Can a model trained on past traffic detect future attacks, possibly of **different types**?”

**XGBoost temporal results (binary class, default threshold):**

- Accuracy: ~0.82
- Precision: ~0.998
- Recall: ~0.28
- F1-score: ~0.44
- AUC-ROC: ~0.80

**Key insight:**

- Excellent **precision** (almost no false alarms)
- Low **recall** (misses most new attacks)
- This is a classic trade-off in security: the model is conservative and very “sure” when it says “attack”, but it is afraid to flag uncertain cases.

---

## 5. Detection by Attack Type (Temporal Split)

Using temporal split, we analyzed detection rate per attack type.

### 5.1 XGBoost-Only Detection Rates

| Attack Type                 | Total | Detected | Detection Rate |
|----------------------------|-------|----------|----------------|
| Web Attack Brute Force     | 1,507 | 1,433    | 95.1%          |
| Web Attack XSS             | 652   | 626      | 96.0%          |
| Web Attack SQL Injection   | 21    | 18       | 85.7%          |
| DDoS                       | 128,027 | 86,800 | 67.8%          |
| PortScan                   | 158,930 | ~888   | ~0.6%          |
| Bot                        | 1,966 | 0        | 0.0%           |
| Infiltration               | 36    | 0        | 0.0%           |
| Benign (correctly benign)  | 871,074 | ~870,555 | ~99.9%       |

### 5.2 Interpretation

- **Web Attacks** generalize very well:
  - Training: FTP/SSH brute-force attacks
  - Testing: Web brute-force, XSS, SQL injection
  - Behavior is similar (repetitive login/HTTP attempts)

- **DDoS** partially generalizes:
  - Training: DoS attacks (Hulk, Slowloris, etc.)
  - Testing: DDoS (distributed version)
  - Different scale and pattern, but still related → moderate detection

- **PortScan, Bot, Infiltration** do not generalize:
  - No similar behavior in training days
  - Port scans: many small flows probing ports
  - Bot: stealthy, blends with normal traffic
  - Infiltration: very few and subtle samples
  - Model has no “experience” with these behaviors → fails to detect

- **False alarms** remain very low:
  - Only a few hundred false positives out of hundreds of thousands of benign flows.

---

## 6. Hybrid Model: XGBoost + Isolation Forest

To address the limitation of supervised-only detection, a hybrid approach was tested:

### 6.1 Combination Strategies

- Weighted average of scores:
  - `hybrid_score = α * xgb_score + (1 - α) * iso_score`
  - Tested with α = 0.7, 0.6, 0.5
- OR logic:
  - Attack if `XGB >= threshold_xgb OR IF >= threshold_if`
- Max score:
  - `max(xgb_score, iso_score)`

### 6.2 Best Hybrid Configuration (Empirical)

- **Weighted (70% XGB + 30% IF)**

Results (temporal):

- F1 ≈ 0.44 (slight improvement over XGBoost-only)
- Precision ≈ 0.996
- Recall ≈ 0.285
- AUC-ROC ≈ 0.80
- False alarms: ~300 / 871,074 ≈ 0.03%

Detection improvements vs XGBoost-only were **small** (a few percentage points at most for some web attacks and DDoS), and **Bot/PortScan remained almost undetected.**

### 6.3 Honest Conclusion on Hybrid Approach

- The hybrid model provides:
  - Slight improvements in **recall** and **AUC**
  - Maintains very good **precision**
- But it **does not fundamentally solve** the problem of:
  - Completely new attack categories (Bot, PortScan)
  - Subtle, rare attacks (Infiltration)

This is a **data limitation**, not purely a model limitation:
- The system never saw PortScan or Bot-like traffic in training days.
- An unsupervised model like Isolation Forest can help, but subtle anomalies are still hard to distinguish from natural variability in benign traffic.

---

## 7. Final Model Choice and Justification

### 7.1 For “Known Attack Patterns”

- **XGBoost** is the primary choice:
  - High AUC and F1 in non-temporal evaluation
  - Good temporal AUC (~0.80) with very high precision
  - Efficient inference for real-time scoring

### 7.2 For “Unknown / Emerging Attacks”

- **Isolation Forest** (or autoencoder) can be used as:
  - A **secondary layer** to catch abnormal patterns
  - A source of “suspicious” alerts that analysts can review

### 7.3 Recommended Deployment Architecture

- **Primary detector**: Tuned XGBoost classifier
- **Secondary anomaly detector**: Isolation Forest
- **Hybrid scoring** (e.g., 70% XGB + 30% IF) used to:
  - Flag high-confidence known attacks
  - Surface potential zero-day-like anomalies
- **Thresholds** must be tuned based on:
  - Acceptable false alarm rate
  - Required detection coverage (recall)

---

## 8. Key Lessons Learned in Phase 3

1. **Perfect metrics can be misleading**:
   - AUC = 1.0 on random splits was heavily influenced by data leakage and dataset characteristics.
2. **Proper evaluation is critical**:
   - Temporal split revealed the **true generalization capacity** of models.
3. **Models learn behaviors, not labels**:
   - When test attacks share behavior with training attacks, generalization is strong (web attacks).
   - When behavior is new (PortScan, Bot), supervised models fail.
4. **Hybrid approaches have potential but limits**:
   - Combining supervised and unsupervised methods is realistic and useful.
   - But detection of very subtle or new attack types remains challenging.
5. **Security context matters**:
   - A model with 99% precision but 30% recall might be acceptable as a **low-noise detector** in a SOC.
   - A high-recall, lower-precision variant could be used in different contexts (e.g., offline analysis).

---

## 9. Artifacts Produced in Phase 3

Code:

- `src/models/base_model.py` — base class for detectors
- `src/models/isolation_forest.py` — Isolation Forest logic
- `src/models/random_forest.py` — Random Forest logic
- `src/models/xgboost_model.py` — XGBoost logic
- `src/models/autoencoder.py` — Keras autoencoder
- `src/models/comparator.py` — model comparison utilities
- `src/models/robust_evaluator.py` — deduplicated / temporal / CV evaluation

Notebooks (main logic):

- `notebooks/03_model_training.ipynb` — initial training & evaluation
- `notebooks/04_leakage_check.ipynb` — leakage investigation
- `notebooks/05_robust_evaluation.ipynb` — robust evaluation strategies
- `notebooks/06_improved_temporal.ipynb` — tuned temporal XGBoost
- `notebooks/07_hybrid_model.ipynb` — hybrid XGB + IF detection

Data artifacts:

- Model-specific metrics and comparison tables (stored under `data/processed/` or `saved_models/`, depending on implementation).

---

## 10. Transition to Phase 4

With trained models and a clear understanding of their behavior, Phase 4 will focus on:

1. **Serving** the best model(s) via an **API** (FastAPI)
2. Building a **dashboard** (e.g., Streamlit) for:
   - Visualizing anomaly scores
   - Inspecting flagged flows
   - Monitoring basic KPIs (attack rates, false alarms)
3. **Containerizing** the solution (Docker) for easy deployment
4. Providing clear **usage documentation** and examples:
   - How to send network features to the API
   - How to interpret model responses

Phase 4 turns NetSentinel from a **research prototype** into a usable **engineering tool**.

---