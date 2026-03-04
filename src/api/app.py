"""
═══════════════════════════════════════════════════════════════
NetSentinel — FastAPI Application
REST API for real-time network anomaly detection
═══════════════════════════════════════════════════════════════
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.api.predictor import NetSentinelPredictor

# ── Initialize App ────────────────────────────────
app = FastAPI(
    title="NetSentinel API",
    description="""
    AI-Based Network Anomaly Detection System.

    This API provides real-time classification of network traffic flows
    as **benign** or **attack** using a hybrid XGBoost + Isolation Forest model.

    **Features:**
    - Single flow prediction
    - Batch prediction
    - Model health check
    """,
    version="1.0.0",
    contact={
        "name": "Souhail Bourhim",
        "url": "https://github.com/SouhailBourhim"
    }
)

# ── Load Models ───────────────────────────────────
predictor = NetSentinelPredictor(models_path="saved_models")


# ── Request / Response Models ─────────────────────

class NetworkFlow(BaseModel):
    """
    A single network flow's features.
    Each field corresponds to a CIC-IDS2017 feature.
    """
    destination_port: float = Field(0, description="Destination port number")
    flow_duration: float = Field(0, description="Flow duration in microseconds")
    total_fwd_packets: float = Field(0, description="Total forward packets")
    total_backward_packets: float = Field(0, description="Total backward packets")
    total_length_of_fwd_packets: float = Field(0, description="Total forward packet bytes")
    total_length_of_bwd_packets: float = Field(0, description="Total backward packet bytes")
    fwd_packet_length_max: float = Field(0, description="Max forward packet length")
    fwd_packet_length_min: float = Field(0, description="Min forward packet length")
    fwd_packet_length_mean: float = Field(0, description="Mean forward packet length")
    fwd_packet_length_std: float = Field(0, description="Std of forward packet length")
    bwd_packet_length_max: float = Field(0, description="Max backward packet length")
    bwd_packet_length_min: float = Field(0, description="Min backward packet length")
    bwd_packet_length_mean: float = Field(0, description="Mean backward packet length")
    bwd_packet_length_std: float = Field(0, description="Std of backward packet length")
    flow_bytes_s: float = Field(0, description="Flow bytes per second")
    flow_packets_s: float = Field(0, description="Flow packets per second")
    flow_iat_mean: float = Field(0, description="Mean flow inter-arrival time")
    flow_iat_std: float = Field(0, description="Std of flow inter-arrival time")
    flow_iat_max: float = Field(0, description="Max flow inter-arrival time")
    flow_iat_min: float = Field(0, description="Min flow inter-arrival time")
    fwd_iat_total: float = Field(0, description="Total forward IAT")
    fwd_iat_mean: float = Field(0, description="Mean forward IAT")
    fwd_iat_std: float = Field(0, description="Std of forward IAT")
    fwd_iat_max: float = Field(0, description="Max forward IAT")
    fwd_iat_min: float = Field(0, description="Min forward IAT")
    bwd_iat_total: float = Field(0, description="Total backward IAT")
    bwd_iat_mean: float = Field(0, description="Mean backward IAT")
    bwd_iat_std: float = Field(0, description="Std of backward IAT")
    bwd_iat_max: float = Field(0, description="Max backward IAT")
    bwd_iat_min: float = Field(0, description="Min backward IAT")
    fwd_psh_flags: float = Field(0, description="Forward PSH flag count")
    fwd_header_length: float = Field(0, description="Forward header length")
    bwd_header_length: float = Field(0, description="Backward header length")
    fwd_packets_s: float = Field(0, description="Forward packets per second")
    bwd_packets_s: float = Field(0, description="Backward packets per second")
    min_packet_length: float = Field(0, description="Minimum packet length")
    max_packet_length: float = Field(0, description="Maximum packet length")
    packet_length_mean: float = Field(0, description="Mean packet length")
    packet_length_std: float = Field(0, description="Std of packet length")
    packet_length_variance: float = Field(0, description="Variance of packet length")
    syn_flag_count: float = Field(0, description="SYN flag count")
    ack_flag_count: float = Field(0, description="ACK flag count")
    down_up_ratio: float = Field(0, description="Download/upload ratio")
    average_packet_size: float = Field(0, description="Average packet size")
    avg_fwd_segment_size: float = Field(0, description="Average forward segment size")
    avg_bwd_segment_size: float = Field(0, description="Average backward segment size")
    init_win_bytes_forward: float = Field(0, description="Initial window bytes (forward)")
    init_win_bytes_backward: float = Field(0, description="Initial window bytes (backward)")
    active_mean: float = Field(0, description="Mean active time")
    idle_mean: float = Field(0, description="Mean idle time")

    class Config:
        json_schema_extra = {
            "example": {
                "destination_port": 80,
                "flow_duration": 120000,
                "total_fwd_packets": 10,
                "total_backward_packets": 8,
                "flow_bytes_s": 5000.0,
                "flow_packets_s": 15.0,
                "syn_flag_count": 1,
                "ack_flag_count": 1
            }
        }


class PredictionResponse(BaseModel):
    """Response for a single prediction."""
    label: str = Field(..., description="'benign' or 'attack'")
    confidence: float = Field(..., description="Confidence score [0, 1]")
    xgb_score: float = Field(..., description="XGBoost attack probability")
    iso_score: float = Field(..., description="Isolation Forest anomaly score")
    hybrid_score: float = Field(..., description="Weighted hybrid score")
    threshold: float = Field(0.5, description="Decision threshold")


class BatchRequest(BaseModel):
    """Request for batch prediction."""
    flows: List[NetworkFlow]


class BatchResponse(BaseModel):
    """Response for batch prediction."""
    predictions: List[PredictionResponse]
    total_flows: int
    attacks_detected: int
    benign_detected: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: dict
    version: str


# ── Endpoints ─────────────────────────────────────

@app.get("/", tags=["General"])
async def root():
    """Welcome endpoint."""
    return {
        "message": "NetSentinel — AI-Based Network Anomaly Detection",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Check if all models are loaded and ready."""
    info = predictor.get_model_info()
    return HealthResponse(
        status="healthy" if info["xgb_loaded"] else "degraded",
        models_loaded=info,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single(flow: NetworkFlow):
    """
    Predict whether a single network flow is benign or an attack.

    Send network flow features and receive a classification result
    with confidence scores from both XGBoost and Isolation Forest models.
    """
    try:
        features = flow.model_dump()
        result = predictor.predict_single(features)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
async def predict_batch(request: BatchRequest):
    """
    Predict on multiple network flows at once.

    Send a list of flows and receive classification results for all.
    """
    try:
        features_list = [flow.model_dump() for flow in request.flows]
        results = predictor.predict_batch(features_list)

        predictions = [PredictionResponse(**r) for r in results]
        attacks = sum(1 for p in predictions if p.label == "attack")

        return BatchResponse(
            predictions=predictions,
            total_flows=len(predictions),
            attacks_detected=attacks,
            benign_detected=len(predictions) - attacks
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get information about the loaded models."""
    return predictor.get_model_info()