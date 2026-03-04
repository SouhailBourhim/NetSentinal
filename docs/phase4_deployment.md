# NetSentinel — Phase 4: Deployment (API, Dashboard, Docker)

---

## 1. Objective

Phase 4 turns NetSentinel from an offline ML project into a **usable tool** by:

1. Providing a **REST API** (FastAPI) for real-time anomaly detection.
2. Providing a **dashboard** (Streamlit) for interactive analysis and monitoring.
3. Packaging everything with **Docker** so it can be run easily on any machine.

This phase assumes that:

- Phase 2 has generated a cleaned dataset.
- Phase 3 has trained and saved models under `saved_models/`.

---

## 2. Deployment Architecture

```text
                ┌──────────────────────────────┐
                │         NetSentinel          │
                │       (Deployed Stack)       │
                └──────────────────────────────┘
                          ▲           ▲
                          │           │
                    HTTP / JSON   HTTP / Web
                          │           │
                    ┌─────┴─────┐ ┌───┴────────┐
                    │  FastAPI  │ │ Streamlit  │
                    │   API     │ │ Dashboard  │
                    │  :8000    │ │   :8501    │
                    └────┬──────┘ └────┬───────┘
                         │             │
                         └──────┬──────┘
                                │
                        ┌───────▼────────┐
                        │  Predictor     │
                        │  Service       │
                        └───────┬────────┘
                                │
         ┌──────────────────────┼──────────────────────┐
         │                      │                      │
   ┌─────▼────┐           ┌─────▼────┐           ┌─────▼────┐
   │XGBoost   │           │IsoForest │           │Scaler+   │
   │(tuned)   │           │(unsup.)  │           │Features  │
   └──────────┘           └──────────┘           └──────────┘

 saved_models/
   ├── xgboost_tuned.pkl
   ├── isolation_forest.pkl
   ├── random_forest.pkl
   ├── autoencoder.keras
   ├── scaler.pkl
   ├── feature_names.json
   └── model_comparison.json
```

## 3. Prediction Service (`src/api/predictor.py`)

### 3.1 Responsibilities

The NetSentinelPredictor class:

- Loads all relevant artifacts from `saved_models/`:
  - Tuned XGBoost model
  - Isolation Forest model
  - StandardScaler
  - Feature names
- Validates and prepares input feature vectors
- Produces predictions using:
  - XGBoost alone
  - Isolation Forest alone
  - A hybrid combination of both

### 3.2 Hybrid Scoring

For a given flow:

- `xgb_score`: probability from XGBoost that the flow is an attack
- `iso_score`: anomaly score from Isolation Forest, normalized to [0, 1]
- `hybrid_score = 0.7 * xgb_score + 0.3 * iso_score`

Decision rule:

- If `hybrid_score ≥ 0.5` → label = "attack"
- Else → label = "benign"

The predictor exposes:

| Method | Description |
|--------|-------------|
| `predict_single(features: dict)` | One flow → one prediction dict |
| `predict_batch(list[dict])` | List of flows → list of predictions |
| `predict_dataframe(df: pd.DataFrame)` | DataFrame → DataFrame with extra prediction columns |
| `get_model_info()` | Returns which models are loaded, feature count, and hybrid weights |
## 4. FastAPI Application (`src/api/app.py`)

### 4.1 Endpoints

The FastAPI app wires the predictor into a REST API.

Available endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Basic info + links to docs & health |
| `/health` | GET | Check if models and scaler are loaded |
| `/model/info` | GET | Detailed model info (feature count, weights, etc.) |
| `/predict` | POST | Predict for a single network flow |
| `/predict/batch` | POST | Predict for a batch of flows |

### 4.2 Example Request / Response

#### Single Flow (`/predict`)

**Request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "destination_port": 80,
    "flow_duration": 120000,
    "total_fwd_packets": 10,
    "total_backward_packets": 8,
    "flow_bytes_s": 5000.0,
    "flow_packets_s": 15.0,
    "syn_flag_count": 1,
    "ack_flag_count": 1
  }'
```

**Response (structure):**

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

#### Batch (`/predict/batch`)

**Request:**

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

**Response (structure):**

```json
{
  "predictions": [
    {"label": "benign", "confidence": 0.87, "xgb_score": 0.12, ...},
    {"label": "attack", "confidence": 0.92, "xgb_score": 0.89, ...}
  ],
  "total_flows": 2,
  "attacks_detected": 1,
  "benign_detected": 1
}
```

### 4.3 Running FastAPI

```bash
cd NetSentinel
source venv/bin/activate

uvicorn src.api.app:app --reload --port 8000
```

Open:
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health
## 5. Streamlit Dashboard (`src/dashboard/app.py`)

The Streamlit app provides an analyst-friendly UI on top of the same predictor.

### 5.1 Pages

#### 🏠 Overview

Shows:
- Number of models loaded
- Number of features
- Hybrid weights (XGBoost vs Isolation Forest)
- Explains system architecture and how the hybrid model works.

#### 🔍 Single Flow Analysis

UI with form fields:
- Destination port, flow duration, packet counts
- Forward/backward mean packet sizes and std
- Flow bytes/s, packets/s, SYN and ACK counts, etc.

On click:
- Model classifies the flow (benign vs attack)
- Shows confidence, individual scores, and a gauge chart for anomaly level.

#### 📊 Batch Analysis

Upload a CSV file containing network flows (same features used in training).

Dashboard:
- Computes predictions for all flows
- Shows summary: total flows, attacks, benign, attack rate
- Displays hybrid score distribution (histogram)
- Lists top-N most suspicious flows
- Provides a download button to get the full annotated CSV

#### 📈 Model Performance

Reads `saved_models/model_comparison.json` if present.

Displays:
- Table of AUC, F1, precision, recall per model
- A bar chart comparing models
- Summarizes key findings from Phase 3 (temporal evaluation).

### 5.2 Running the Dashboard

```bash
cd NetSentinel
source venv/bin/activate

streamlit run src/dashboard/app.py
```

Open: http://localhost:8501

## 6. Docker Deployment

### 6.1 Dockerfile

The Dockerfile:

- Uses `python:3.11-slim`
- Installs dependencies from `requirements.txt`
- Copies:
  - `src/`
  - `configs/`
  - `saved_models/` (so the container has the trained models)
- Exposes:
  - 8000 (API)
  - 8501 (Dashboard)
- Defaults to starting FastAPI (API); dashboard is started via docker-compose.

### 6.2 docker-compose.yml

Defines two services:

| Service | Description | Port |
|---------|-------------|------|
| api | FastAPI app (src/api/app.py) | 8000 → host:8000 |
| dashboard | Streamlit app (src/dashboard/app.py) | 8501 → host:8501 |

Both services:
- Are built from the same Dockerfile.
- Mount `./saved_models` into `/app/saved_models` inside the container.

### 6.3 Running with Docker

```bash
cd NetSentinel
docker-compose up --build
```

This will:
- Build the image once
- Start both services

Access:
- API: http://localhost:8000/docs
- Dashboard: http://localhost:8501

To stop:

```bash
Ctrl+C
docker-compose down
```
## 7. Input Features and Preprocessing

The API and dashboard assume the same features and scaling as used in Phase 3:

- Features are numeric, derived from CIC-IDS2017 fields:
  - Ports, packet counts, bytes, time intervals (IAT), header lengths, packet sizes, TCP flags, etc.
- Preprocessing steps (re-used at prediction time):
  - Missing features → filled with 0
  - Columns ordered according to `feature_names.json`
  - Infinite values replaced, NaNs filled
  - Standardization using the saved `scaler.pkl`

This ensures consistency between training and inference.

## 8. How to Rebuild Models and Deploy

If you want to re-train models and re-deploy:

1. **Phase 2** — run `02_preprocessing.ipynb` to rebuild `processed_traffic.csv` and splits.
2. **Phase 3** — run training notebooks to:
   - Train XGBoost, RF, IF, autoencoder
   - Evaluate and select best models
   - Save artifacts to `saved_models/`:
     - `xgboost_tuned.pkl`
     - `isolation_forest.pkl`
     - `scaler.pkl`
     - `feature_names.json`
     - (optional) `model_comparison.json`
3. **Phase 4** — restart API and dashboard (or rebuild Docker image).
## 9. Limitations & Future Work

### Dataset bias:
- CIC-IDS2017 is synthetic and has strong patterns;
- Real traffic may be noisier and more complex.

### Detection gaps:
- Temporal evaluation showed difficulty detecting:
  - Botnet traffic
  - Port scans
  - Infiltration
- Additional training data and more specialized features would be needed.

### Scalability:
- Current deployment targets a single machine;
- For production, consider:
  - Kubernetes
  - Load balancing
  - Integration with SIEM/SOC tools.

### Potential improvements:

- Add authentication and RBAC on the API.
- Integrate with a message bus (e.g., Kafka) for streaming traffic.
- Use time-series context (LSTMs, transformers) instead of single-flow models.
- Implement continuous retraining with real network logs.
## 10. Summary

Phase 4 completes the NetSentinel project by delivering:

- A FastAPI-based REST API for real-time anomaly detection.
- A Streamlit dashboard for analysts to explore and understand model outputs.
- A Dockerized stack that can be launched with a single command.

This makes NetSentinel not just an academic experiment, but a deployable engineering artifact demonstrating:

- Good ML engineering practices
- Realistic evaluation and awareness of limitations
- Clear separation between data, models, and deployment

---

Document generated as part of the NetSentinel project.
Author: Souhail Bourhim
Date: 2025