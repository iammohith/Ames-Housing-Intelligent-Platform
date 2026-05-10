"""
Page 1 — Pipeline Monitor
Live DAG visualization, agent status, real-time metrics, and diagnostics.
"""

import json
import os
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://orchestration-api:8000")

from theme import apply_theme
apply_theme()

# ── Page-level styles ────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* Agent status cards */
.agent-card {
    background: #FFFFFF;
    border: 1px solid #F1F5F9;
    border-radius: 14px;
    padding: 1.2rem;
    margin: 0;
    box-shadow: 0 2px 12px rgba(15, 23, 42, 0.03);
    transition: all 0.2s ease;
}
.agent-card:hover {
    box-shadow: 0 6px 20px rgba(15, 23, 42, 0.06);
    transform: translateY(-1px);
}
.agent-card-idle   { border-left: 4px solid #CBD5E1; }
.agent-card-run    { border-left: 4px solid #3B82F6; box-shadow: 0 4px 16px rgba(59,130,246,0.15); }
.agent-card-ok     { border-left: 4px solid #10B981; }
.agent-card-fail   { border-left: 4px solid #EF4444; background:#FEF2F2; }
.agent-name  { font-weight:700; font-size:0.9rem; color:#0F172A; margin-bottom: 2px; }
.agent-state { font-size:0.75rem; font-weight:600; color:#64748B; text-transform:uppercase; letter-spacing:0.06em; }

/* DAG nodes */
.dag-node {
    display:inline-flex; align-items:center; justify-content:center;
    padding:10px 24px; border-radius:10px; font-size:0.85rem; font-weight:600;
    min-width:130px; letter-spacing:0.02em; font-family:'Outfit',sans-serif;
    transition: all 0.2s ease;
}
.dag-idle    { background:#F8FAFC; border:1px solid #E2E8F0; color:#64748B; }
.dag-run     { background:linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%); border:1px solid #93C5FD; color:#1D4ED8;
               box-shadow:0 0 16px rgba(59,130,246,0.2); animation:pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
.dag-ok      { background:linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%); border:1px solid #6EE7B7; color:#065F46; }
.dag-fail    { background:linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%); border:1px solid #FECACA; color:#B91C1C; }

.dag-wrap    { display:flex; flex-direction:column; align-items:center; gap:0; padding: 12px 0; }
.dag-vrow    { display:flex; justify-content:center; align-items:center; }
.dag-vline   { width:2px; height:30px; background:#E2E8F0; }
.dag-vline-ok{ background:#10B981; }
.dag-vline-run{ background:#3B82F6; }
.dag-fork    { display:flex; gap:90px; align-items:flex-start; justify-content:center; }
.dag-fcol    { display:flex; flex-direction:column; align-items:center; gap:0; }

@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.6} }

/* Log lines */
.log-line { font-family:'JetBrains Mono',monospace; font-size:0.8rem; padding:3px 8px; border-radius:4px; line-height:1.6; }
.log-SUCCESS { color:#059669; } .log-STARTED { color:#3B82F6; }
.log-PROGRESS { color:#475569; } .log-FAILED { color:#DC2626; }
.log-WARNING { color:#D97706; } .log-RETRYING { color:#EA580C; }

/* Prevent metric label truncation */
[data-testid="stMetricLabel"] {
    white-space: normal !important;
    word-break: break-word !important;
}
[data-testid="stMetricLabel"] > div {
    white-space: normal !important;
    overflow: visible !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("run_id", None),
    ("pipeline_status", "IDLE"),
    ("agent_statuses", {}),
    ("metrics", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="page-header">
    <div class="page-header-title">📡 Pipeline Monitor</div>
    <div class="page-header-sub">Trigger and observe the 8-agent ML pipeline in real-time</div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Control Row ───────────────────────────────────────────────────────────────
btn_col, reset_col, status_col = st.columns([2, 1, 3], gap="medium")

with btn_col:
    if st.button("▶  Run Pipeline", type="primary", use_container_width=True):
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
                st.session_state.agent_statuses = {}
                st.session_state.metrics = {}
                st.success(f"Pipeline started — Run ID: `{data['run_id']}`")
            else:
                st.error(f"Failed to start pipeline: {resp.text}")
        except Exception as e:
            st.error(f"Cannot reach API: {e}")

with reset_col:
    if st.button("↺  Reset", use_container_width=True):
        st.session_state.run_id = None
        st.session_state.pipeline_status = "IDLE"
        st.session_state.agent_statuses = {}
        st.session_state.metrics = {}
        st.rerun()

with status_col:
    colour = {"IDLE": "#94A3B8", "RUNNING": "#3B82F6", "SUCCESS": "#10B981", "FAILED": "#EF4444"}.get(
        st.session_state.pipeline_status, "#94A3B8"
    )
    run_label = st.session_state.run_id or "—"
    st.markdown(
        f"""
<div style="display:flex;align-items:center;gap:10px;padding:0.7rem 1rem;
            background:#fff;border:1px solid #E2E8F0;border-radius:8px;margin-top:2px;">
    <div style="width:10px;height:10px;border-radius:50%;background:{colour};flex-shrink:0;"></div>
    <div>
        <span style="font-weight:700;font-size:0.9rem;color:#0F172A;">{st.session_state.pipeline_status}</span>
        <span style="color:#64748B;font-size:0.82rem;margin-left:10px;">Run: <code style="background:#F1F5F9;padding:1px 5px;border-radius:4px;">{run_label}</code></span>
    </div>
</div>""",
        unsafe_allow_html=True,
    )

# ── Poll API ──────────────────────────────────────────────────────────────────
if st.session_state.run_id:
    try:
        resp = requests.get(f"{API_URL}/api/status/{st.session_state.run_id}", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            st.session_state.agent_statuses = data.get("agents", {})
            st.session_state.metrics = data.get("metrics", {})
            progress = data.get("progress_pct", 0)
            if data.get("status") in ("SUCCESS", "FAILED"):
                st.session_state.pipeline_status = data["status"]
            if st.session_state.pipeline_status == "RUNNING":
                st.progress(progress / 100.0, text=f"Pipeline progress: {progress:.0f}%")
    except Exception:
        pass
elif not st.session_state.metrics:
    try:
        r = requests.get(f"{API_URL}/api/latest-metrics", timeout=4)
        if r.status_code == 200:
            st.session_state.metrics = r.json().get("metrics", {})
    except Exception:
        pass

st.markdown("<br>", unsafe_allow_html=True)

# ── DAG ───────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Live DAG Execution</div>', unsafe_allow_html=True)

AGENTS_SEQ = ["ingestion_agent", "schema_agent", "cleaning_agent", "feature_agent", "encoding_agent"]
AGENTS_PARALLEL = ["anomaly_agent", "ml_agent"]
AGENT_LABELS = {
    "ingestion_agent": "INGESTION",
    "schema_agent": "SCHEMA",
    "cleaning_agent": "CLEANING",
    "feature_agent": "FEATURES",
    "encoding_agent": "ENCODING",
    "anomaly_agent": "ANOMALY",
    "ml_agent": "ML TRAINING",
    "orchestration_agent": "ORCHESTRATION",
}


def _dag_cls(agent_id):
    s = st.session_state.agent_statuses.get(agent_id, "IDLE")
    if s == "SUCCESS":
        return "dag-ok"
    if s in ("STARTED", "PROGRESS"):
        return "dag-run"
    if s == "FAILED":
        return "dag-fail"
    return "dag-idle"


def _vline_cls(agent_id):
    s = st.session_state.agent_statuses.get(agent_id, "IDLE")
    if s == "SUCCESS":
        return "dag-vline dag-vline-ok"
    if s in ("STARTED", "PROGRESS"):
        return "dag-vline dag-vline-run"
    return "dag-vline"


def _node(agent_id):
    label = AGENT_LABELS[agent_id]
    return f'<div class="dag-node {_dag_cls(agent_id)}">{label}</div>'


def _conn(agent_id):
    return f'<div class="{_vline_cls(agent_id)}"></div>'


enc_done_cls = _vline_cls("encoding_agent")

dag_html = f"""
<div class="dag-wrap">
  {_node('ingestion_agent')}{_conn('ingestion_agent')}
  {_node('schema_agent')}{_conn('schema_agent')}
  {_node('cleaning_agent')}{_conn('cleaning_agent')}
  {_node('feature_agent')}{_conn('feature_agent')}
  {_node('encoding_agent')}
  <div class="{enc_done_cls}"></div>
  <div class="dag-fork">
    <div class="dag-fcol">
      <div class="{enc_done_cls}"></div>
      {_node('anomaly_agent')}
      <div class="{_vline_cls('anomaly_agent')}"></div>
    </div>
    <div class="dag-fcol">
      <div class="{enc_done_cls}"></div>
      {_node('ml_agent')}
      <div class="{_vline_cls('ml_agent')}"></div>
    </div>
  </div>
  {_node('orchestration_agent')}
</div>
"""
st.markdown(dag_html, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Agent Status Cards ────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Agent Status</div>', unsafe_allow_html=True)

all_agents = AGENTS_SEQ + AGENTS_PARALLEL + ["orchestration_agent"]
icon_map = {"SUCCESS": "✓", "STARTED": "◌", "PROGRESS": "◌", "FAILED": "✗", "WARNING": "⚠", "RETRYING": "↺"}
cls_map = {"SUCCESS": "agent-card-ok", "STARTED": "agent-card-run", "PROGRESS": "agent-card-run",
           "FAILED": "agent-card-fail"}

cols = st.columns(4, gap="medium")
for i, agent_id in enumerate(all_agents):
    status = st.session_state.agent_statuses.get(agent_id, "IDLE")
    icon = icon_map.get(status, "·")
    card_cls = cls_map.get(status, "agent-card-idle")
    label = AGENT_LABELS[agent_id]
    with cols[i % 4]:
        st.markdown(
            f"""<div class="agent-card {card_cls}">
  <div class="agent-name">{icon} {label}</div>
  <div class="agent-state">{status}</div>
</div>""",
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ── Metrics Row ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Real-Time Metrics</div>', unsafe_allow_html=True)

metrics = st.session_state.get("metrics", {})
m = st.columns([1.3, 1, 1, 1, 1, 1], gap="medium")
m[0].metric("Rows Processed", metrics.get("rows_processed") or "—")
m[1].metric("Features", metrics.get("features_count") or "—")
m[2].metric("Anomalies", metrics.get("anomalies_count") or "—")
m[3].metric("Best RMSE", f"${metrics['best_rmse']:,.0f}" if metrics.get("best_rmse") else "—")
m[4].metric("Best R²", f"{metrics['best_r2']:.3f}" if metrics.get("best_r2") else "—")
m[5].metric("KB Chunks", metrics.get("knowledge_chunks") or "—")

st.markdown("<br>", unsafe_allow_html=True)

# ── Diagnostics Tabs ──────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Diagnostics</div>', unsafe_allow_html=True)

if st.session_state.pipeline_status == "RUNNING":
    st.info("⏳ Diagnostics and historical logs are paused while the pipeline is actively running. They will refresh upon completion.")
else:
    tab1, tab2, tab3, tab4 = st.tabs(["Run History", "Data Quality", "Anomaly Log", "Schema Drift"])

    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        try:
            resp = requests.get(f"{API_URL}/api/pipeline-runs", timeout=3)
            if resp.status_code == 200:
                runs = resp.json().get("runs", [])
                if runs:
                    df_runs = pd.DataFrame(runs)
                    df_runs["started_at"] = pd.to_datetime(df_runs["started_at"])
                    df_runs = df_runs.sort_values("started_at")
                    df_runs["duration_ms"] = df_runs.get("duration_ms", pd.Series([0] * len(df_runs))).fillna(0)

                    fig = px.bar(
                        df_runs, x="started_at", y="duration_ms", color="status",
                        color_discrete_map={"SUCCESS": "#10B981", "FAILED": "#EF4444", "RUNNING": "#3B82F6"},
                        labels={"started_at": "Run Timestamp", "duration_ms": "Duration (ms)"},
                        template="plotly_white",
                    )
                    fig.update_layout(
                        height=320, margin=dict(l=0, r=0, t=20, b=0), font=dict(family="Inter"),
                        showlegend=True, legend=dict(orientation="h", y=1.08),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(
                        df_runs.sort_values("started_at", ascending=False),
                        use_container_width=True, hide_index=True,
                    )
                else:
                    st.info("No pipeline runs recorded yet. Run the pipeline to populate this view.")
        except Exception:
            st.info("Run the pipeline to populate history.")

    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        try:
            resp = requests.get(f"{API_URL}/api/schema-history", timeout=3)
            if resp.status_code == 200:
                history = resp.json().get("history", [])
                if history:
                    import numpy as np

                    df_hist = pd.DataFrame(history)
                    latest_run = df_hist.iloc[0]["run_id"]
                    df_latest = df_hist[df_hist["run_id"] == latest_run].copy()
                    if "is_structural_na" not in df_latest.columns:
                        df_latest["is_structural_na"] = df_latest["null_rate"] > 0.4
                    df_sorted = df_latest.sort_values("null_rate", ascending=False)
                    df_with_nulls = df_sorted[df_sorted["null_rate"] > 0].head(30)
                    if df_with_nulls.empty:
                        df_with_nulls = df_sorted.head(30)
                    columns = df_with_nulls["column"].tolist()
                    null_rates = df_with_nulls["null_rate"].tolist()
                    structural = [1.0 if v else 0.0 for v in df_with_nulls["is_structural_na"].tolist()]
                    dtypes = df_with_nulls["data_type"].tolist()
                    hover_null = [f"{c}<br>Null Rate: {nr:.1%}<br>Type: {dt}" for c, nr, dt in zip(columns, null_rates, dtypes)]
                    hover_struct = [f"{c}<br>Structural NA: {'Yes' if s else 'No'}" for c, s in zip(columns, structural)]
                    fig_hm = go.Figure(data=go.Heatmap(
                        z=[null_rates, structural], x=columns,
                        y=["Null Rate", "Structural NA"],
                        colorscale=[[0, "#ECFDF5"], [0.2, "#A7F3D0"], [0.4, "#FEF3C7"],
                                    [0.6, "#FBBF24"], [0.8, "#F87171"], [1, "#B91C1C"]],
                        hovertext=[hover_null, hover_struct], hoverinfo="text",
                        showscale=True,
                        colorbar=dict(title="Rate", tickformat=".0%", thickness=12, len=0.8),
                        zmin=0, zmax=1,
                    ))
                    fig_hm.update_layout(
                        height=260, margin=dict(l=0, r=0, t=16, b=0),
                        font=dict(family="Inter", size=11),
                        xaxis=dict(tickangle=-40, side="bottom"),
                        yaxis=dict(autorange="reversed"),
                        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                    )
                    st.plotly_chart(fig_hm, use_container_width=True)
                    st.dataframe(
                        df_latest[["column", "data_type", "null_rate", "is_structural_na"]]
                        .sort_values("null_rate", ascending=False).set_index("column"),
                        column_config={"null_rate": st.column_config.ProgressColumn(
                            "Null Rate", format="%.2f", min_value=0, max_value=1)},
                        use_container_width=True, height=300,
                    )
                else:
                    st.info("No data quality metrics yet.")
        except Exception:
            st.info("Run the pipeline to populate data quality metrics.")

    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)
        try:
            resp = requests.get(f"{API_URL}/api/anomalies", timeout=3)
            if resp.status_code == 200:
                anomalies = resp.json().get("anomalies", [])
                if anomalies:
                    df_anom = pd.DataFrame(anomalies)
                    c_tbl, c_detail = st.columns([1, 1], gap="large")
                    with c_tbl:
                        st.dataframe(
                            df_anom[["pid", "neighborhood", "severity", "isolation_score", "methods"]],
                            use_container_width=True, hide_index=True, height=380,
                        )
                    with c_detail:
                        pids = df_anom["pid"].astype(str).tolist()
                        selected_pid = st.selectbox("Select property for detail", pids)
                        if selected_pid:
                            row = df_anom[df_anom["pid"].astype(str) == selected_pid].iloc[0]
                            st.markdown(f"**PID {row['pid']}** · {row['neighborhood']}")
                            st.markdown(
                                f"Severity: `{row['severity']}` &nbsp;|&nbsp; Isolation score: `{row.get('isolation_score','—')}`",
                                unsafe_allow_html=True,
                            )
                            features = row.get("anomalous_features", "{}")
                            if isinstance(features, str):
                                try:
                                    features = json.loads(features)
                                except Exception:
                                    features = {}
                            st.json(features)
                else:
                    st.info("No anomalies recorded yet.")
        except Exception:
            st.info("Run the pipeline to populate anomaly logs.")

    with tab4:
        st.markdown("<br>", unsafe_allow_html=True)
        try:
            resp = requests.get(f"{API_URL}/api/schema-history", timeout=3)
            if resp.status_code == 200:
                history = resp.json().get("history", [])
                if history:
                    df_hist = pd.DataFrame(history)
                    top_cols = (
                        df_hist.groupby("column")["null_rate"].max()
                        .sort_values(ascending=False).head(5).index
                    )
                    df_plot = df_hist[df_hist["column"].isin(top_cols)].sort_values("run_id")
                    fig = px.line(
                        df_plot, x="run_id", y="null_rate", color="column", markers=True,
                        labels={"run_id": "Pipeline Run", "null_rate": "Null Rate", "column": "Feature"},
                        template="plotly_white",
                    )
                    fig.update_layout(height=360, margin=dict(l=0, r=0, t=20, b=0), font=dict(family="Inter"))
                    fig.layout.yaxis.tickformat = ".0%"
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No schema drift history recorded yet.")
        except Exception:
            st.info("Run the pipeline to populate schema drift history.")

# ── Auto-refresh during run ───────────────────────────────────────────────────
if st.session_state.pipeline_status == "RUNNING":
    time.sleep(3)
    st.rerun()
