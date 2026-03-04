# 🛡️ NetSentinel — AI-Based Network Anomaly Detection System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36-red)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📋 Overview

**NetSentinel** is an end-to-end AI-powered network anomaly detection system
that classifies network traffic flows as **benign** or **attack** using a
hybrid machine learning approach combining:

- 🎯 **XGBoost** — Supervised classifier for known attack patterns
- 🔍 **Isolation Forest** — Unsupervised anomaly detector for unknown threats
- ⚡ **Hybrid Scoring** — Weighted combination for optimal detection

The system includes a **REST API** (FastAPI), a **monitoring dashboard**
(Streamlit), and **Docker** containerization for easy deployment.

---

## 🏗️ Architecture
┌─────────────────────────────────────────────────────────┐
│ NetSentinel │
│ │
│ Network Traffic │
│ │ │
│ ▼ │
│ ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│ │ Feature │───▶│ Scaler │───▶│ Hybrid Model │ │
│ │ Extraction│ │(Standard)│ │ │ │
│ └──────────┘ └──────────┘ │ ┌──────┐ ┌─────┐│ │
│ │ │XGBoost│ │ IF ││ │
│ │ │ (70%) │ │(30%)││ │
│ │ └──┬───┘ └──┬──┘│ │
│ │ └────┬───┘ │ │
│ │ ▼ │ │
│ │ Hybrid Score │ │
│ └────────┬────────┘ │
│ │ │
│ ┌─────────────┴──────────┐ │
│ │ │ │
│ ┌────▼────┐ ┌──────▼─┐│
│ │ FastAPI │ │Streamlit││
│ │ :8000 │ │ :8501 ││
│ └─────────┘ └────────┘│
└─────────────────────────────────────────────────────────┘


---

## 📁 Project Structure
NetSentinel/
├── README.md # This file
├── requirements.txt # Python dependencies
├── Dockerfile # Container definition
├── docker-compose.yml # Multi-service orchestration
│
├── configs/
│ └── model_config.yaml # Model hyperparameters
│
├── data/
│ ├── raw/ # CIC-IDS2017 CSV files (not tracked)
│ └── processed/ # Cleaned & engineered data (not tracked)
│ └── cleaning_report.json # Data cleaning documentation
│
├── docs/
│ ├── phase1_exploration.md # Phase 1 documentation
│ ├── phase2_preprocessing.md # Phase 2 documentation
│ ├── phase3_models.md # Phase 3 documentation
│ └── phase4_deployment.md # Phase 4 documentation
│
├── notebooks/
│ ├── 01_exploration.ipynb # Data exploration & EDA
│ ├── 02_preprocessing.ipynb # Cleaning & feature engineering
│ ├── 03_model_training.ipynb # Baseline model training
│ ├── 04_leakage_check.ipynb # Data leakage investigation
│ ├── 05_robust_evaluation.ipynb # Multi-strategy evaluation
│ ├── 06_improved_temporal.ipynb # Temporal XGBoost optimization
│ ├── 07_hybrid_model.ipynb # Hybrid XGB + IF approach
│ └── 08_phase3_completion.ipynb # Final training & model saving
│
├── saved_models/ # Trained model artifacts (not tracked)
│ ├── xgboost_tuned.pkl
│ ├── isolation_forest.pkl
│ ├── random_forest.pkl
│ ├── autoencoder.keras
│ ├── scaler.pkl
│ ├── feature_names.json
│ ├── best_params.json
│ └── model_comparison.json
│
└── src/
├── data/
│ ├── preprocessor.py # Data cleaning pipeline
│ └── splitter.py # Train/test split + SMOTE
│
├── features/
│ └── engineer.py # Feature engineering
│
├── models/
│ ├── base_model.py # Abstract detector interface
│ ├── isolation_forest.py # Unsupervised detector
│ ├── random_forest.py # Supervised detector
│ ├── xgboost_model.py # Gradient boosting detector
│ ├── autoencoder.py # Deep learning detector
│ ├── comparator.py # Model comparison utilities
│ └── robust_evaluator.py # Leakage-aware evaluation
│
├── api/
│ ├── predictor.py # Model loading & prediction
│ └── app.py # FastAPI application
│
└── dashboard/
└── app.py # Streamlit dashboard


---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- pip
- Docker (optional, for containerized deployment)

### 1. Clone the Repository

```bash
git clone https://github.com/SouhailBourhim/NetSentinel.git
cd NetSentinel
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate    # macOS/Linux
# venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Dataset

Download CIC-IDS2017 (MachineLearningCSV.zip) from:
https://www.unb.ca/cic/datasets/ids-2017.html

Extract CSV files into data/raw/:
```bash
mkdir -p data/raw
# Extract CSVs into data/raw/
```

### 5. Run the Pipeline

```bash
# Step 1: Explore data
jupyter notebook notebooks/01_exploration.ipynb

# Step 2: Preprocess
jupyter notebook notebooks/02_preprocessing.ipynb

# Step 3: Train models
jupyter notebook notebooks/03_model_training.ipynb

# Steps 4-8: Evaluation & analysis
jupyter notebook notebooks/04_leakage_check.ipynb
jupyter notebook notebooks/05_robust_evaluation.ipynb
# ... etc.
```

### 6. Start the API

```bash
uvicorn src.api.app:app --reload --port 8000
```
Open API docs: http://localhost:8000/docs

### 7. Start the Dashboard

```bash
streamlit run src/dashboard/app.py
```
Open dashboard: http://localhost:8501

### 8. Run with Docker

```bash
docker-compose up --build
```
API: http://localhost:8000/docs
Dashboard: http://localhost:8501

---

## 🔌 API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "models_loaded": {
    "xgb_loaded": true,
    "iso_loaded": true,
    "scaler_loaded": true,
    "feature_count": 48
  },
  "version": "1.0.0"
}
```

### Single Flow Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "destination_port": 80,
    "flow_duration": 120000,
    "total_fwd_packets": 10,
    "total_backward_packets": 8,
    "flow_bytes_s": 5000.0,
    "syn_flag_count": 1,
    "ack_flag_count": 1
  }'
```

Response:
```json
{
  "label": "benign",
  "confidence": 0.87,
  "xgb_score": 0.12,
  "iso_score": 0.15,
  "hybrid_score": 0.13,
  "threshold": 0.5
}
```

### Batch Prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "flows": [
      {"destination_port": 80, "flow_duration": 120000, "total_fwd_packets": 10},
      {"destination_port": 22, "flow_duration": 500, "total_fwd_packets": 1000}
    ]
  }'
```
## 📊 Model Performance

### Evaluation Summary

| Strategy | AUC | F1 | Precision | Recall |
|----------|-----|----|-----------|--------|
| Standard split (leaked) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Deduplicated split | 1.0000 | 0.9978 | 0.9969 | 0.9986 |
| Cross-validation | 1.0000 | 0.9959 | 0.9951 | 0.9967 |
| Temporal split (realistic) | 0.8015 | 0.4429 | 0.9964 | 0.2847 |

### Detection by Attack Type (Temporal Split)

| Attack Type | Detection Rate | Status |
|-------------|----------------|--------|
| Web Attack Brute Force | 95.1% | ✅ High |
| Web Attack XSS | 96.0% | ✅ High |
| Web Attack SQL Injection | 85.7% | ✅ High |
| DDoS | 67.8% | 🟡 Moderate |
| PortScan | 0.6% | 🔴 Low |
| Bot | 0.0% | 🔴 Undetected |
| Infiltration | 0.0% | 🔴 Undetected |

### Key Insights

- Perfect AUC on standard splits was inflated by 73% near-duplicate leakage in CIC-IDS2017.
- Temporal evaluation reveals realistic generalization (AUC ≈ 0.80).
- Web attacks generalize well because training included similar brute-force patterns.
- Novel attack types (Bot, PortScan) require retraining with new data.
- False alarm rate is excellent (0.03%) — operationally viable.
## 🧠 Technical Highlights

### Data Pipeline

- Cleaned 2.8M+ network flows (NaN, Inf, duplicates)
- Engineered 12 domain-specific features (ratios, entropy, behavioral)
- Reduced from 79 to 48 features via correlation analysis
- Stratified split + StandardScaler + SMOTE balancing

### Models

- 4 models implemented: Isolation Forest, Random Forest, XGBoost, Autoencoder
- Hybrid approach: 70% XGBoost + 30% Isolation Forest
- Hyperparameter tuning via GridSearchCV

### Evaluation Rigor

- Detected and quantified data leakage (73% near-duplicates)
- 4 evaluation strategies: standard, deduplicated, cross-validation, temporal
- Per-attack-type detection analysis
- Honest reporting of model limitations

### Deployment

- FastAPI REST API with Swagger documentation
- Streamlit interactive dashboard with batch analysis
- Docker containerization with docker-compose
## 📖 Documentation

Detailed documentation for each phase:

| Phase | Document | Description |
|-------|----------|-------------|
| Phase 1 | docs/phase1_exploration.md | Data acquisition & EDA |
| Phase 2 | docs/phase2_preprocessing.md | Preprocessing & feature engineering |
| Phase 3 | docs/phase3_models.md | Model training & evaluation |
| Phase 4 | docs/phase4_deployment.md | API, dashboard & Docker |
## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| Language | Python 3.11 |
| ML/DL | XGBoost, Scikit-learn, TensorFlow/Keras |
| Data | Pandas, NumPy, imbalanced-learn (SMOTE) |
| API | FastAPI, Uvicorn, Pydantic |
| Dashboard | Streamlit, Plotly |
| Containerization | Docker, Docker Compose |
| Visualization | Matplotlib, Seaborn, Plotly |
| Dataset | CIC-IDS2017 (University of New Brunswick) |
## 👨‍💻 Author

**Souhail Bourhim**

Engineering Student at INPT (Smart-ICT)

- [GitHub](https://github.com/SouhailBourhim)
- [LinkedIn](https://linkedin.com/in/souhailbourhim)
## 📄 License

This project is licensed under the MIT License.

## � Acknowledgments

- **CIC-IDS2017 Dataset** — Canadian Institute for Cybersecurity, UNB
- **INPT** — Institut National des Postes et Télécommunications
- Inspired by real-world telecom security architectures
