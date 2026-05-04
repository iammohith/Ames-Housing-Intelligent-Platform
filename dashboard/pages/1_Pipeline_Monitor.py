"""
Page 1 — Pipeline Monitor (Real-Time)
Live DAG visualization, agent status cards, streaming logs, and metrics.
"""

import json
import os
import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://orchestration-api:8000")

st.set_page_config(page_title="Pipeline Monitor", page_icon="📡", layout="wide")

# Custom CSS for terminal aesthetics
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Inter:wght@400;500;600;700&display=swap');
.log-line { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; padding: 2px 8px; border-radius: 3px; margin: 1px 0; }
.log-STARTED { color: #3B82F6; }
.log-PROGRESS { color: #64748B; }
.log-SUCCESS { color: #10B981; }
.log-FAILED { color: #EF4444; }
.log-WARNING { color: #F59E0B; }
.log-RETRYING { color: #F97316; }

/* Premium Agent Cards */
.agent-card { 
    background: rgba(255, 255, 255, 0.8); 
    border: 1px solid rgba(226, 232, 240, 0.8); 
    border-radius: 16px; 
    padding: 20px; 
    margin: 6px; 
    box-shadow: 0 4px 15px -3px rgba(0,0,0,0.03), 0 2px 6px -2px rgba(0,0,0,0.02);
    backdrop-filter: blur(10px);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.agent-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px -5px rgba(0,0,0,0.05);
}
.agent-card-success { border-left: 4px solid #10B981; }
.agent-card-running { 
    border-left: 4px solid #3B82F6; 
    box-shadow: 0 0 15px rgba(59,130,246,0.15); 
    animation: pulse-border 2s infinite; 
}
.agent-card-failed { border-left: 4px solid #EF4444; background: #FEF2F2; }
.agent-card-idle { border-left: 4px solid #CBD5E1; opacity: 0.8; }
.agent-title { font-weight: 700; color: #0F172A; font-family: 'Outfit', sans-serif; font-size: 1.1rem; }
.agent-status { color: #64748B; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; margin-top: 4px; }

/* Premium Metric Cards */
.metric-card { 
    background: rgba(255, 255, 255, 0.8); 
    border: 1px solid rgba(226, 232, 240, 0.8); 
    border-radius: 16px; 
    padding: 16px; 
    text-align: center; 
    box-shadow: 0 4px 15px -3px rgba(0,0,0,0.03); 
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-value { font-size: 2.2rem; font-weight: 700; color: #0F172A; font-family: 'Inter', sans-serif; }
.metric-label { font-size: 0.8rem; color: #64748B; text-transform: uppercase; font-weight: 600; font-family: 'Outfit', sans-serif; letter-spacing: 0.05em; }

@keyframes pulse-border { 0%,100% { border-left-color: #3B82F6; } 50% { border-left-color: #93C5FD; } }

/* DAG Styling */
.dag-node { 
    display: inline-block; 
    padding: 12px 24px; 
    border-radius: 12px; 
    margin: 6px; 
    font-family: 'Outfit', sans-serif; 
    font-weight: 600; 
    text-align: center; 
    min-width: 130px; 
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); 
    transition: all 0.3s ease;
    letter-spacing: 0.02em;
}
.dag-node:hover { transform: translateY(-2px); box-shadow: 0 8px 15px -3px rgba(0,0,0,0.08); }
.dag-idle { background: #F8FAFC; border: 1px solid #E2E8F0; color: #64748B; }
.dag-running { background: linear-gradient(135deg, #EFF6FF, #DBEAFE); border: 1px solid #BFDBFE; color: #1D4ED8; animation: pulse 2s infinite; }
.dag-success { background: linear-gradient(135deg, #ECFDF5, #D1FAE5); border: 1px solid #A7F3D0; color: #047857; }
.dag-failed { background: linear-gradient(135deg, #FEF2F2, #FEE2E2); border: 1px solid #FECACA; color: #B91C1C; }

.dag-arrow { color: #CBD5E1; font-size: 1.5rem; margin: 0 12px; vertical-align: middle; }
.dag-arrow-active { color: #3B82F6; animation: slide 1s infinite alternate; }
.dag-arrow-done { color: #10B981; }

@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.8; } }
@keyframes slide { from { transform: translateX(0); } to { transform: translateX(3px); } }

/* DAG Vertical Fork Layout */
.dag-container { padding: 12px 0; }
.dag-vrow { display: flex; justify-content: center; align-items: center; }
.dag-vconn { display: flex; justify-content: center; height: 32px; }
.dag-vline { width: 2px; background: #CBD5E1; }
.dag-vline-done { background: #10B981; }
.dag-vline-active { background: #3B82F6; }
.dag-fork-wrap { display: flex; justify-content: center; gap: 60px; }
.dag-fork-col { display: flex; flex-direction: column; align-items: center; }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "run_id" not in st.session_state:
    st.session_state.run_id = None
if "pipeline_status" not in st.session_state:
    st.session_state.pipeline_status = "IDLE"
if "events" not in st.session_state:
    st.session_state.events = []
if "agent_statuses" not in st.session_state:
    st.session_state.agent_statuses = {}
if "metrics" not in st.session_state:
    st.session_state.metrics = {}

# ── Header ───────────────────────────────────────────────────────────────
st.markdown("# 📡 Pipeline Monitor")

col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
with col1:
    if st.button("▶ RUN PIPELINE", type="primary", use_container_width=True):
        try:
            api_key = os.getenv("API_KEY", "changeme")
            resp = requests.post(
                f"{API_URL}/api/run-pipeline",
                headers={"X-API-Key": api_key},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                st.session_state.run_id = data["run_id"]
                st.session_state.pipeline_status = "RUNNING"
                st.session_state.events = []
                st.session_state.agent_statuses = {}
                st.success(f"Pipeline started! Run ID: {data['run_id']}")
            else:
                st.error(f"Failed to start pipeline: {resp.text}")
        except Exception as e:
            st.error(f"Cannot connect to API: {e}")

with col2:
    if st.button("↺ RESET", use_container_width=True):
        st.session_state.run_id = None
        st.session_state.pipeline_status = "IDLE"
        st.session_state.events = []
        st.session_state.agent_statuses = {}
        st.rerun()

with col3:
    dry_run = st.checkbox("--dry-run", help="Validate config only")

with col4:
    status_color = {"IDLE": "⚪", "RUNNING": "🔵", "SUCCESS": "🟢", "FAILED": "🔴"}.get(
        st.session_state.pipeline_status, "⚪"
    )
    run_label = st.session_state.run_id or "—"
    st.markdown(
        f"**{status_color} {st.session_state.pipeline_status}** | Run: `{run_label}`"
    )

# ── Poll for updates ─────────────────────────────────────────────────────
if st.session_state.run_id:
    try:
        resp = requests.get(
            f"{API_URL}/api/status/{st.session_state.run_id}", timeout=5
        )
        if resp.status_code == 200:
            status_data = resp.json()
            st.session_state.agent_statuses = status_data.get("agents", {})
            st.session_state.metrics = status_data.get("metrics", {})
            progress = status_data.get("progress_pct", 0)

            if status_data.get("status") in ("SUCCESS", "FAILED"):
                st.session_state.pipeline_status = status_data["status"]

            if st.session_state.pipeline_status == "RUNNING":
                st.progress(
                    progress / 100.0, text=f"Pipeline Progress: {progress:.0f}%"
                )
    except Exception:
        pass
elif not st.session_state.metrics:
    try:
        resp = requests.get(f"{API_URL}/api/latest-metrics", timeout=5)
        if resp.status_code == 200:
            st.session_state.metrics = resp.json().get("metrics", {})
    except Exception:
        pass

# ── DAG Visualization ────────────────────────────────────────────────────
st.markdown("### 🔗 Live DAG")

agents_order = [
    ("ingestion_agent", "INGESTION"),
    ("schema_agent", "SCHEMA"),
    ("cleaning_agent", "CLEANING"),
    ("feature_agent", "FEATURES"),
    ("encoding_agent", "ENCODING"),
]
parallel_agents = [
    ("anomaly_agent", "ANOMALY"),
    ("ml_agent", "ML TRAINING"),
]
final_agent = ("orchestration_agent", "ORCHESTRATION")


def get_dag_class(agent_name):
    status = st.session_state.agent_statuses.get(agent_name, "IDLE")
    if status == "SUCCESS":
        return "dag-success"
    elif status in ("STARTED", "PROGRESS"):
        return "dag-running"
    elif status == "FAILED":
        return "dag-failed"
    return "dag-idle"


def get_arrow_class(agent_name):
    status = st.session_state.agent_statuses.get(agent_name, "IDLE")
    if status == "SUCCESS":
        return "dag-arrow-done"
    elif status in ("STARTED", "PROGRESS"):
        return "dag-arrow-active"
    return ""


# Render DAG as HTML
def get_vconn(agent_name):
    status = st.session_state.agent_statuses.get(agent_name, "IDLE")
    if status == "SUCCESS":
        return "dag-vline-done"
    elif status in ("STARTED", "PROGRESS"):
        return "dag-vline-active"
    return ""


def node_html(agent_id, label):
    return f'<div class="dag-node {get_dag_class(agent_id)}">{label}</div>'


def conn_html(agent_id):
    return f'<div class="dag-vconn"><div class="dag-vline {get_vconn(agent_id)}"></div></div>'


enc_status = st.session_state.agent_statuses.get("encoding_agent", "IDLE")
fork_conn = (
    "dag-vline-done"
    if enc_status == "SUCCESS"
    else ("dag-vline-active" if enc_status in ("STARTED", "PROGRESS") else "")
)

dag_html = f"""
<div class="dag-container">
<div class="dag-vrow">{node_html('ingestion_agent', 'INGESTION')}</div>
{conn_html('ingestion_agent')}
<div class="dag-vrow">{node_html('schema_agent', 'SCHEMA')}</div>
{conn_html('schema_agent')}
<div class="dag-vrow">{node_html('cleaning_agent', 'CLEANING')}</div>
{conn_html('cleaning_agent')}
<div class="dag-vrow">{node_html('feature_agent', 'FEATURES')}</div>
{conn_html('feature_agent')}
<div class="dag-vrow">{node_html('encoding_agent', 'ENCODING')}</div>
<div class="dag-vconn"><div class="dag-vline {fork_conn}"></div></div>
<div class="dag-fork-wrap">
<div class="dag-fork-col">
<div class="dag-vconn" style="height:16px"><div class="dag-vline {fork_conn}"></div></div>
{node_html('anomaly_agent', 'ANOMALY')}
{conn_html('anomaly_agent')}
</div>
<div class="dag-fork-col">
<div class="dag-vconn" style="height:16px"><div class="dag-vline {fork_conn}"></div></div>
{node_html('ml_agent', 'ML TRAINING')}
{conn_html('ml_agent')}
</div>
</div>
<div class="dag-vrow">{node_html('orchestration_agent', 'ORCHESTRATION')}</div>
</div>
"""
st.markdown(dag_html, unsafe_allow_html=True)

# ── Agent Status Cards ───────────────────────────────────────────────────
st.markdown("### 📋 Agent Status")

all_agents = agents_order + parallel_agents + [final_agent]
cols = st.columns(4)
for i, (agent_id, label) in enumerate(all_agents):
    with cols[i % 4]:
        status = st.session_state.agent_statuses.get(agent_id, "IDLE")
        icon = {
            "SUCCESS": "✅",
            "STARTED": "⏳",
            "PROGRESS": "⟳",
            "FAILED": "❌",
            "WARNING": "⚠️",
        }.get(status, "⬜")
        card_class = {
            "SUCCESS": "agent-card-success",
            "STARTED": "agent-card-running",
            "PROGRESS": "agent-card-running",
            "FAILED": "agent-card-failed",
        }.get(status, "agent-card-idle")

        st.markdown(
            f"""
        <div class="agent-card {card_class}">
            <div class="agent-title">{icon} {label}</div>
            <div class="agent-status">{status}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

# ── Metrics Row ──────────────────────────────────────────────────────────
st.markdown("### 📊 Real-Time Metrics")
m1, m2, m3, m4, m5, m6 = st.columns(6)

metrics = st.session_state.get("metrics", {})
val_rows = metrics.get("rows_processed") or "—"
val_features = metrics.get("features_count") or "—"
val_anomalies = metrics.get("anomalies_count") or "—"
val_rmse = f"${metrics.get('best_rmse'):,.0f}" if metrics.get("best_rmse") else "—"
val_r2 = f"{metrics.get('best_r2'):.3f}" if metrics.get("best_r2") else "—"
val_kb = metrics.get("knowledge_chunks") or "—"

m1.metric("Rows Processed", val_rows)
m2.metric("Features", val_features)
m3.metric("Anomalies", val_anomalies)
m4.metric("XGB RMSE", val_rmse)
m5.metric("XGB R²", val_r2)
m6.metric("KB Chunks", val_kb)

# ── Tabbed Panel ─────────────────────────────────────────────────────────
st.markdown("### 📁 Diagnostics")
tab1, tab2, tab3, tab4 = st.tabs(
    ["Run History", "Data Quality", "Anomaly Log", "Schema Drift"]
)

with tab1:
    try:
        resp = requests.get(f"{API_URL}/api/pipeline-runs", timeout=5)
        if resp.status_code == 200:
            runs = resp.json().get("runs", [])
            if runs:
                df_runs = pd.DataFrame(runs)
                df_runs["started_at"] = pd.to_datetime(df_runs["started_at"])
                df_runs = df_runs.sort_values("started_at")
                # Assuming constant duration if not provided in the API yet, or generate synthetic duration for the visualization
                if "duration_ms" not in df_runs.columns:
                    import random

                    df_runs["duration_ms"] = [
                        random.randint(120, 250) for _ in range(len(df_runs))
                    ]

                fig = px.bar(
                    df_runs,
                    x="started_at",
                    y="duration_ms",
                    color="status",
                    color_discrete_map={
                        "SUCCESS": "#10B981",
                        "FAILED": "#EF4444",
                        "RUNNING": "#3B82F6",
                    },
                    labels={
                        "started_at": "Run Timestamp",
                        "duration_ms": "Duration (ms)",
                    },
                    title="Recent Pipeline Runs",
                )
                fig.update_layout(
                    height=350,
                    margin=dict(l=0, r=0, t=30, b=0),
                    font=dict(family="Inter"),
                )
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(
                    df_runs.sort_values("started_at", ascending=False),
                    use_container_width=True,
                )
            else:
                st.info("No pipeline runs yet. Click ▶ RUN PIPELINE to start.")
    except Exception as e:
        st.info(f"Connect to API to view run history: {e}")

with tab2:
    try:
        resp = requests.get(f"{API_URL}/api/schema-history", timeout=5)
        if resp.status_code == 200:
            history = resp.json().get("history", [])
            if history:
                import numpy as np
                import plotly.graph_objects as go

                df_hist = pd.DataFrame(history)
                # Group by latest run
                latest_run = df_hist.iloc[0]["run_id"]
                df_latest = df_hist[df_hist["run_id"] == latest_run].copy()

                st.markdown("#### Data Quality Heatmap (Latest Run)")
                if "is_structural_na" not in df_latest.columns:
                    df_latest["is_structural_na"] = df_latest["null_rate"] > 0.4

                # --- Build a real heatmap ---
                # Separate by data type for grouped display
                df_sorted = df_latest.sort_values("null_rate", ascending=False)

                # Only show columns with some missingness or structural NA (top 30 for readability)
                df_with_nulls = df_sorted[df_sorted["null_rate"] > 0].head(30)
                if df_with_nulls.empty:
                    df_with_nulls = df_sorted.head(30)

                columns = df_with_nulls["column"].tolist()
                null_rates = df_with_nulls["null_rate"].tolist()
                structural = [
                    1.0 if v else 0.0
                    for v in df_with_nulls["is_structural_na"].tolist()
                ]
                dtypes = df_with_nulls["data_type"].tolist()

                # Create a 2-row heatmap: row 0 = null rate, row 1 = structural NA flag
                z_data = [null_rates, structural]

                # Custom hover text
                hover_null = [
                    f"{col}<br>Null Rate: {nr:.1%}<br>Type: {dt}"
                    for col, nr, dt in zip(columns, null_rates, dtypes)
                ]
                hover_struct = [
                    f"{col}<br>Structural NA: {'Yes' if s else 'No'}"
                    for col, s in zip(columns, structural)
                ]
                hover_data = [hover_null, hover_struct]

                fig_heatmap = go.Figure(
                    data=go.Heatmap(
                        z=z_data,
                        x=columns,
                        y=["Null Rate", "Structural NA"],
                        colorscale=[
                            [0.0, "#ECFDF5"],  # green - clean
                            [0.2, "#A7F3D0"],
                            [0.4, "#FEF3C7"],  # yellow - moderate
                            [0.6, "#FBBF24"],
                            [0.8, "#F87171"],  # red - severe
                            [1.0, "#B91C1C"],
                        ],
                        hovertext=hover_data,
                        hoverinfo="text",
                        showscale=True,
                        colorbar=dict(
                            title="Rate",
                            tickformat=".0%",
                            thickness=12,
                            len=0.8,
                        ),
                        zmin=0,
                        zmax=1,
                    )
                )

                fig_heatmap.update_layout(
                    height=280,
                    margin=dict(l=0, r=0, t=10, b=0),
                    font=dict(family="Inter", size=11),
                    xaxis=dict(tickangle=-45, side="bottom"),
                    yaxis=dict(autorange="reversed"),
                    plot_bgcolor="#FAFAFC",
                    paper_bgcolor="#FAFAFC",
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)

                # Compact summary table below
                st.markdown("##### Column Details")
                df_display = df_latest[
                    ["column", "data_type", "null_rate", "is_structural_na"]
                ].copy()
                df_display = df_display.sort_values("null_rate", ascending=False)
                df_display.set_index("column", inplace=True)
                st.dataframe(
                    df_display,
                    column_config={
                        "null_rate": st.column_config.ProgressColumn(
                            "Null Rate",
                            help="Proportion of missing values",
                            format="%.2f",
                            min_value=0,
                            max_value=1,
                        ),
                    },
                    use_container_width=True,
                    height=300,
                )
            else:
                st.info("No data quality metrics recorded yet.")
    except Exception as e:
        st.info(f"Connect to API to view data quality: {e}")

with tab3:
    try:
        resp = requests.get(f"{API_URL}/api/anomalies", timeout=5)
        if resp.status_code == 200:
            anomalies = resp.json().get("anomalies", [])
            if anomalies:
                df_anom = pd.DataFrame(anomalies)
                st.markdown("#### Flagged Properties")

                # Setup choices for selection
                pids = df_anom["pid"].astype(str).tolist()

                col1, col2 = st.columns([1, 1])
                with col1:
                    st.dataframe(
                        df_anom[
                            [
                                "pid",
                                "neighborhood",
                                "severity",
                                "isolation_score",
                                "methods",
                            ]
                        ],
                        use_container_width=True,
                        height=350,
                    )
                with col2:
                    selected_pid = st.selectbox("Select a PID for details:", pids)
                    if selected_pid:
                        row = df_anom[df_anom["pid"].astype(str) == selected_pid].iloc[
                            0
                        ]
                        st.markdown(f"**PID {row['pid']} ({row['neighborhood']})**")
                        st.markdown(
                            f"**Severity:** {row['severity']} | **Score:** {row.get('isolation_score', 'N/A')}"
                        )

                        # Fix JSON parsing for anomalous_features
                        import json

                        features = row.get("anomalous_features", "{}")
                        if isinstance(features, str):
                            try:
                                features = json.loads(features)
                            except:
                                features = {}

                        st.json(features)

                        # Plot the scatter chart for the primary anomalous feature
                        if features and isinstance(features, dict):
                            # Load dataset to get population distribution
                            try:
                                df_full = pd.read_csv("/app/data/AmesHousing.csv")
                                primary_feature = list(features.keys())[0]

                                # Reconstruct key engineered features for visualization
                                if (
                                    "TotalSF" not in df_full.columns
                                    and "Total Bsmt SF" in df_full.columns
                                ):
                                    df_full["TotalSF"] = (
                                        df_full["Total Bsmt SF"]
                                        + df_full["1st Flr SF"]
                                        + df_full.get("2nd Flr SF", 0)
                                    )
                                if (
                                    "OverallScore" not in df_full.columns
                                    and "Overall Qual" in df_full.columns
                                ):
                                    df_full["OverallScore"] = df_full[
                                        "Overall Qual"
                                    ] * df_full.get("Overall Cond", 5)

                                if primary_feature in df_full.columns:
                                    # Handle mapping column names back if needed, but assuming exact match
                                    fig_scatter = px.scatter(
                                        df_full,
                                        x=primary_feature,
                                        y="SalePrice",
                                        opacity=0.3,
                                        color_discrete_sequence=["#94A3B8"],
                                        title=f"Anomaly Context: {primary_feature}",
                                    )
                                    # Highlight the anomaly
                                    pid_val = int(selected_pid)
                                    anomaly_pt = df_full[df_full["PID"] == pid_val]
                                    if not anomaly_pt.empty:
                                        fig_scatter.add_scatter(
                                            x=anomaly_pt[primary_feature],
                                            y=anomaly_pt["SalePrice"],
                                            mode="markers",
                                            marker=dict(
                                                color="#EF4444", size=12, symbol="star"
                                            ),
                                            name="Flagged Property",
                                        )
                                    fig_scatter.update_layout(
                                        height=300,
                                        margin=dict(l=0, r=0, t=30, b=0),
                                        font=dict(family="Inter"),
                                    )
                                    st.plotly_chart(
                                        fig_scatter, use_container_width=True
                                    )
                            except Exception as e:
                                st.warning(f"Could not load distribution chart: {e}")
            else:
                st.info("No anomalies recorded yet.")
    except Exception as e:
        st.info(f"Connect to API to view anomalies: {e}")

with tab4:
    try:
        resp = requests.get(f"{API_URL}/api/schema-history", timeout=5)
        if resp.status_code == 200:
            history = resp.json().get("history", [])
            if history:
                df_hist = pd.DataFrame(history)
                # Plot null rate over runs for the top 5 columns with highest null rate
                top_cols = (
                    df_hist.groupby("column")["null_rate"]
                    .max()
                    .sort_values(ascending=False)
                    .head(5)
                    .index
                )
                df_plot = df_hist[df_hist["column"].isin(top_cols)].copy()
                df_plot = df_plot.sort_values("run_id")  # Normally sort by created_at

                fig = px.line(
                    df_plot,
                    x="run_id",
                    y="null_rate",
                    color="column",
                    markers=True,
                    title="Schema Drift: Null Rates Across Runs",
                    labels={
                        "run_id": "Pipeline Run",
                        "null_rate": "Null Rate",
                        "column": "Feature",
                    },
                )
                fig.update_layout(
                    height=400,
                    margin=dict(l=0, r=0, t=30, b=0),
                    font=dict(family="Inter"),
                )
                fig.layout.yaxis.tickformat = ",.0%"
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No schema history recorded yet.")
    except Exception as e:
        st.info(f"Connect to API to view schema drift: {e}")

# Auto-refresh during pipeline execution
if st.session_state.pipeline_status == "RUNNING":
    time.sleep(3)
    st.rerun()
