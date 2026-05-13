-- Ames Housing Intelligence Platform — PostgreSQL Schema
-- Auto-executed on first container startup via docker-entrypoint-initdb.d

-- ── Pipeline Runs ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id          VARCHAR(64) PRIMARY KEY,
    dataset_hash    VARCHAR(128) NOT NULL,
    status          VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    started_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMP WITH TIME ZONE,
    duration_ms     INTEGER,
    rows_in         INTEGER,
    rows_out        INTEGER,
    config_snapshot JSONB,
    error_message   TEXT,
    created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_pipeline_runs_status ON pipeline_runs(status);
CREATE INDEX idx_pipeline_runs_hash ON pipeline_runs(dataset_hash);

-- ── Agent Runs (per-agent execution records) ───────────────────────────────
CREATE TABLE IF NOT EXISTS agent_runs (
    id              SERIAL PRIMARY KEY,
    run_id          VARCHAR(64) NOT NULL REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
    agent_name      VARCHAR(50) NOT NULL,
    status          VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    started_at      TIMESTAMP WITH TIME ZONE,
    completed_at    TIMESTAMP WITH TIME ZONE,
    duration_ms     INTEGER,
    rows_in         INTEGER,
    rows_out        INTEGER,
    metadata        JSONB,
    error_message   TEXT,
    traceback       TEXT,
    created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_agent_runs_run_id ON agent_runs(run_id);
CREATE INDEX idx_agent_runs_agent ON agent_runs(agent_name);

-- ── Schema History (per-column metrics across runs for drift detection) ────
CREATE TABLE IF NOT EXISTS schema_history (
    id              SERIAL PRIMARY KEY,
    run_id          VARCHAR(64) NOT NULL REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
    column_name     VARCHAR(100) NOT NULL,
    data_type       VARCHAR(50),
    null_rate       FLOAT NOT NULL DEFAULT 0.0,
    unique_count    INTEGER,
    is_structural_na BOOLEAN NOT NULL DEFAULT FALSE,
    stats           JSONB,
    created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_schema_history_run ON schema_history(run_id);
CREATE INDEX idx_schema_history_col ON schema_history(column_name);

-- ── Anomaly Log ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS anomaly_log (
    id              SERIAL PRIMARY KEY,
    run_id          VARCHAR(64) NOT NULL REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
    pid             VARCHAR(100),
    neighborhood    VARCHAR(50),
    methods         TEXT[] NOT NULL DEFAULT '{}',
    isolation_score FLOAT,
    severity        VARCHAR(10) NOT NULL DEFAULT 'LOW',
    anomalous_features JSONB,
    details         JSONB,
    created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_anomaly_log_run ON anomaly_log(run_id);
CREATE INDEX idx_anomaly_log_severity ON anomaly_log(severity);

-- ── Cleaning Log ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS cleaning_log (
    id              SERIAL PRIMARY KEY,
    run_id          VARCHAR(64) NOT NULL REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
    column_name     VARCHAR(100) NOT NULL,
    action          VARCHAR(50) NOT NULL,
    rows_affected   INTEGER NOT NULL DEFAULT 0,
    method          VARCHAR(100),
    details         JSONB,
    created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_cleaning_log_run ON cleaning_log(run_id);

-- ── Model Results ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS model_results (
    id              SERIAL PRIMARY KEY,
    run_id          VARCHAR(64) NOT NULL REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
    model_name      VARCHAR(50) NOT NULL,
    val_rmse        FLOAT,
    val_r2          FLOAT,
    test_rmse       FLOAT,
    test_r2         FLOAT,
    test_mae        FLOAT,
    test_mape       FLOAT,
    rmse_log        FLOAT,
    is_best         BOOLEAN NOT NULL DEFAULT FALSE,
    hyperparameters JSONB,
    mlflow_run_id   VARCHAR(64),
    created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_model_results_run ON model_results(run_id);
CREATE INDEX idx_model_results_best ON model_results(is_best) WHERE is_best = TRUE;
