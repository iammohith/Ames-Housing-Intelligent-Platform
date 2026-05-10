"""
Prometheus metrics — all counters, histograms, and gauges for the platform.
"""

from prometheus_client import Counter, Gauge, Histogram

# ── Counters ─────────────────────────────────────────────────────────────────

pipeline_runs_total = Counter(
    "pipeline_runs_total",
    "Total pipeline executions",
    ["status"],
)

agent_runs_total = Counter(
    "agent_runs_total",
    "Total agent executions",
    ["agent_name", "status"],
)

# ── Histograms ───────────────────────────────────────────────────────────────

agent_duration_seconds = Histogram(
    "agent_duration_seconds",
    "Agent execution time in seconds",
    ["agent_name"],
    buckets=(0.5, 1, 2, 5, 10, 30, 60, 120, 300),
)

rag_query_duration_seconds = Histogram(
    "rag_query_duration_seconds",
    "End-to-end RAG query latency in seconds",
    buckets=(0.1, 0.5, 1, 2, 5, 10),
)

api_request_duration_seconds = Histogram(
    "api_request_duration_seconds",
    "API request duration in seconds",
    ["endpoint", "method", "status_code"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 5),
)

# ── Gauges ───────────────────────────────────────────────────────────────────

anomalies_detected_total = Gauge(
    "anomalies_detected_total",
    "Number of anomalies detected in last run",
)

model_rmse = Gauge(
    "model_rmse",
    "Model RMSE",
    ["model_name", "split"],
)

model_r2 = Gauge(
    "model_r2",
    "Model R-squared",
    ["model_name", "split"],
)

model_mae = Gauge(
    "model_mae",
    "Model MAE",
    ["model_name", "split"],
)

data_drift_score = Gauge(
    "data_drift_score",
    "Data drift score per column (null rate delta vs baseline)",
    ["column_name"],
)

knowledge_base_chunks_total = Gauge(
    "knowledge_base_chunks_total",
    "Total chunks in ChromaDB knowledge base",
)

pipeline_currently_running = Gauge(
    "pipeline_currently_running",
    "Whether a pipeline is currently running (0 or 1)",
)

rows_processed_last_run = Gauge(
    "rows_processed_last_run",
    "Number of rows processed in the most recent run",
)

# ── Initialize label combinations so metrics are always visible to Prometheus ──

_AGENT_NAMES = [
    "ingestion_agent",
    "schema_agent",
    "cleaning_agent",
    "feature_agent",
    "encoding_agent",
    "anomaly_agent",
    "ml_agent",
    "orchestration_agent",
]

# Pre-register counter label combos (value stays 0 until incremented)
pipeline_runs_total.labels(status="success")
pipeline_runs_total.labels(status="failure")

for _agent in _AGENT_NAMES:
    agent_runs_total.labels(agent_name=_agent, status="success")
    agent_runs_total.labels(agent_name=_agent, status="failure")
    agent_duration_seconds.labels(agent_name=_agent)

# Pre-register gauge metrics so they appear in Prometheus before first value is set
_MODEL_NAMES = ["ridge", "xgboost", "lightgbm"]
_SPLITS = ["val", "test"]
for _model in _MODEL_NAMES:
    for _split in _SPLITS:
        model_rmse.labels(model_name=_model, split=_split).set(0)
        model_r2.labels(model_name=_model, split=_split).set(0)
        model_mae.labels(model_name=_model, split=_split).set(0)

# Initialize gauges with 0
anomalies_detected_total.set(0)
knowledge_base_chunks_total.set(0)
pipeline_currently_running.set(0)
rows_processed_last_run.set(0)
