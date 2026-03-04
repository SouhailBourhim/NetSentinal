"""
═══════════════════════════════════════════════════════════════
NetSentinel — Streamlit Dashboard
Real-time network anomaly monitoring interface
═══════════════════════════════════════════════════════════════
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import sys
import os

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.api.predictor import NetSentinelPredictor

# ── Page Config ───────────────────────────────────
st.set_page_config(
    page_title="NetSentinel — Network Anomaly Detection",
    page_icon="🛡️",
    layout="wide"
)

# ── Load Predictor ────────────────────────────────
@st.cache_resource
def load_predictor():
    return NetSentinelPredictor(models_path="saved_models")

predictor = load_predictor()


# ── Sidebar ───────────────────────────────────────
st.sidebar.title("🛡️ NetSentinel")
st.sidebar.markdown("**AI-Based Network Anomaly Detection**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "🔍 Single Flow Analysis", "📊 Batch Analysis", "📈 Model Performance"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Status**")
info = predictor.get_model_info()
if info["xgb_loaded"]:
    st.sidebar.success("XGBoost: ✅ Loaded")
else:
    st.sidebar.error("XGBoost: ❌ Not loaded")

if info["iso_loaded"]:
    st.sidebar.success("Isolation Forest: ✅ Loaded")
else:
    st.sidebar.error("Isolation Forest: ❌ Not loaded")
    
    st.sidebar.info(f"Features: {info['feature_count']}")


# ── Page: Overview ────────────────────────────────
if page == "🏠 Overview":
    st.title("🛡️ NetSentinel — Network Anomaly Detection")
    st.markdown("""
    **NetSentinel** is an AI-powered system for detecting malicious network traffic
    in real-time using a hybrid approach combining:

    - 🎯 **XGBoost** — Supervised classification for known attack patterns
    - 🔍 **Isolation Forest** — Unsupervised anomaly detection for unknown threats
    - ⚡ **Hybrid Scoring** — Weighted combination for optimal detection

    ---
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Models Loaded", "2" if info["xgb_loaded"] and info["iso_loaded"] else "1")
    with col2:
        st.metric("Features Used", info["feature_count"])
    with col3:
        st.metric("XGB Weight", f"{info['hybrid_weights']['xgb']:.0%}")
    with col4:
        st.metric("IF Weight", f"{info['hybrid_weights']['iso']:.0%}")

    st.markdown("---")
    st.markdown("### How It Works")

    st.markdown("""
    ```
    Network Traffic → Feature Extraction → Scaling → Hybrid Model → Alert
                                                        │
                                                  ┌─────┴─────┐
                                                  │           │
                                              XGBoost    Isolation
                                             (known      Forest
                                             attacks)   (anomalies)
                                                  │           │
                                                  └─────┬─────┘
                                                        │
                                                  Weighted Score
                                                        │
                                                  Benign / Attack
    ```
    """)


# ── Page: Single Flow ────────────────────────────
elif page == "🔍 Single Flow Analysis":
    st.title("🔍 Single Flow Analysis")
    st.markdown("Enter network flow features to get a prediction.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Flow Basics**")
        dest_port = st.number_input("Destination Port", value=80, min_value=0, max_value=65535)
        flow_duration = st.number_input("Flow Duration (μs)", value=120000, min_value=0)
        total_fwd = st.number_input("Total Fwd Packets", value=10, min_value=0)
        total_bwd = st.number_input("Total Bwd Packets", value=8, min_value=0)

    with col2:
        st.markdown("**Packet Sizes**")
        fwd_len_mean = st.number_input("Fwd Packet Length Mean", value=200.0)
        fwd_len_std = st.number_input("Fwd Packet Length Std", value=50.0)
        bwd_len_mean = st.number_input("Bwd Packet Length Mean", value=150.0)
        bwd_len_std = st.number_input("Bwd Packet Length Std", value=40.0)

    with col3:
        st.markdown("**Rates & Flags**")
        flow_bytes = st.number_input("Flow Bytes/s", value=5000.0)
        flow_packets = st.number_input("Flow Packets/s", value=15.0)
        syn_flags = st.number_input("SYN Flag Count", value=1, min_value=0)
        ack_flags = st.number_input("ACK Flag Count", value=1, min_value=0)

    if st.button("🔎 Analyze Flow", type="primary"):
        features = {
            "destination_port": dest_port,
            "flow_duration": flow_duration,
            "total_fwd_packets": total_fwd,
            "total_backward_packets": total_bwd,
            "fwd_packet_length_mean": fwd_len_mean,
            "fwd_packet_length_std": fwd_len_std,
            "bwd_packet_length_mean": bwd_len_mean,
            "bwd_packet_length_std": bwd_len_std,
            "flow_bytes_s": flow_bytes,
            "flow_packets_s": flow_packets,
            "syn_flag_count": syn_flags,
            "ack_flag_count": ack_flags
        }

        result = predictor.predict_single(features)

        st.markdown("---")

        # Result display
        if result["label"] == "attack":
            st.error(f"⚠️ ATTACK DETECTED — Confidence: {result['confidence']:.1%}")
        else:
            st.success(f"✅ BENIGN TRAFFIC — Confidence: {result['confidence']:.1%}")

        # Score breakdown
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("XGBoost Score", f"{result['xgb_score']:.4f}")
        with col2:
            st.metric("Isolation Forest Score", f"{result['iso_score']:.4f}")
        with col3:
            st.metric("Hybrid Score", f"{result['hybrid_score']:.4f}")

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result['hybrid_score'] * 100,
            title={'text': "Anomaly Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#E8683F" if result['label'] == 'attack' else "#2E86AB"},
                'steps': [
                    {'range': [0, 30], 'color': '#E8F5E9'},
                    {'range': [30, 70], 'color': '#FFF3E0'},
                    {'range': [70, 100], 'color': '#FFEBEE'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'value': 50
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)


# ── Page: Batch Analysis ──────────────────────────
elif page == "📊 Batch Analysis":
    st.title("📊 Batch Analysis")
    st.markdown("Upload a CSV file with network flow features for bulk analysis.")

    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, low_memory=False)
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('/', '_').str.lower()

        st.markdown(f"**Loaded:** {len(df):,} flows, {len(df.columns)} features")

        if st.button("🚀 Analyze All Flows", type="primary"):
            with st.spinner("Analyzing flows..."):
                results_df = predictor.predict_dataframe(df)

            # Summary
            st.markdown("---")
            st.markdown("### Results Summary")

            total = len(results_df)
            attacks = (results_df['prediction'] == 'attack').sum()
            benign = total - attacks

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Flows", f"{total:,}")
            with col2:
                st.metric("Benign", f"{benign:,}")
            with col3:
                st.metric("Attacks", f"{attacks:,}")
            with col4:
                st.metric("Attack Rate", f"{attacks/total*100:.2f}%")

            # Distribution chart
            fig = px.histogram(
                results_df,
                x='hybrid_score',
                color='prediction',
                nbins=100,
                color_discrete_map={'benign': '#2E86AB', 'attack': '#E8683F'},
                title="Hybrid Score Distribution"
            )
            fig.add_vline(x=0.5, line_dash="dash", line_color="red",
                         annotation_text="Threshold")
            st.plotly_chart(fig, use_container_width=True)

            # Top suspicious flows
            st.markdown("### Top 20 Most Suspicious Flows")
            suspicious = results_df.nlargest(20, 'hybrid_score')
            display_cols = ['prediction', 'hybrid_score', 'xgb_score', 'iso_score', 'confidence']
            extra_cols = [c for c in ['destination_port', 'flow_duration',
                         'total_fwd_packets', 'flow_bytes_s'] if c in suspicious.columns]
            st.dataframe(suspicious[display_cols + extra_cols], use_container_width=True)

            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                "📥 Download Full Results (CSV)",
                csv,
                "netsentinel_results.csv",
                "text/csv"
            )


# ── Page: Model Performance ──────────────────────
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    st.markdown("Overview of model evaluation results from Phase 3.")

    # Load comparison data if available
    comparison_path = "saved_models/model_comparison.json"
    if os.path.exists(comparison_path):
        with open(comparison_path, "r") as f:
            comparison = json.load(f)

        st.markdown("### Temporal Evaluation Results (Train Mon-Wed → Test Thu-Fri)")

        # Table
        df_comp = pd.DataFrame(comparison).T
        st.dataframe(df_comp.style.format("{:.4f}"), use_container_width=True)

        # Bar chart
        metrics = ['f1', 'precision', 'recall', 'auc']
        available_metrics = [m for m in metrics if m in df_comp.columns]

        if available_metrics:
            fig = px.bar(
                df_comp.reset_index().melt(
                    id_vars='index',
                    value_vars=available_metrics
                ),
                x='index',
                y='value',
                color='variable',
                barmode='group',
                title="Model Comparison",
                labels={'index': 'Model', 'value': 'Score', 'variable': 'Metric'}
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No model comparison data found. Run Phase 3 evaluation first.")

    st.markdown("---")
    st.markdown("### Key Findings")
    st.markdown("""
    - **Standard evaluation** (random split) showed AUC ≈ 1.0, but this was
      inflated by near-duplicate leakage in the CIC-IDS2017 dataset.
    - **Temporal evaluation** (train Mon-Wed, test Thu-Fri) provides realistic
      performance estimates (AUC ≈ 0.80).
    - **Web attacks** generalize well across time periods (>85% detection).
    - **Novel attack types** (Bot, PortScan) remain challenging without
      retraining on new data.
    - **Hybrid approach** (XGBoost + Isolation Forest) provides marginal
      improvement over XGBoost alone.
    """)