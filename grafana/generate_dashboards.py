#!/usr/bin/env python3
"""
Programmatic Grafana Dashboard Generator
Generates 3 provisioned dashboards as JSON files.
Run during Docker build or manually: python generate_dashboards.py
"""

import json
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "dashboards")


def _panel(title, panel_type, gridPos, targets, overrides=None, options=None):
    """Helper to build a Grafana panel dict."""
    panel = {
        "title": title,
        "type": panel_type,
        "gridPos": gridPos,
        "datasource": {"type": "prometheus", "uid": "prometheus"},
        "targets": targets,
        "fieldConfig": {"defaults": {}, "overrides": overrides or []},
        "options": options or {},
    }
    return panel


def _prom_target(expr, legend="", ref="A"):
    return {
        "expr": expr,
        "legendFormat": legend,
        "refId": ref,
        "datasource": {"type": "prometheus", "uid": "prometheus"},
    }


def generate_pipeline_health():
    """Dashboard 1: Pipeline Health — run success rate, per-agent latency."""
    panels = [
        _panel(
            "Pipeline Runs — Success vs Failure (24h)",
            "stat",
            {"h": 6, "w": 6, "x": 0, "y": 0},
            [
                _prom_target(
                    'sum(increase(pipeline_runs_total{status="success"}[24h]))',
                    "Success",
                    "A",
                ),
                _prom_target(
                    'sum(increase(pipeline_runs_total{status="failure"}[24h]))',
                    "Failure",
                    "B",
                ),
            ],
        ),
        _panel(
            "Pipeline Currently Running",
            "stat",
            {"h": 6, "w": 4, "x": 6, "y": 0},
            [_prom_target("pipeline_currently_running", "Running")],
        ),
        _panel(
            "Rows Processed (Last Run)",
            "stat",
            {"h": 6, "w": 4, "x": 10, "y": 0},
            [_prom_target("rows_processed_last_run", "Rows")],
        ),
        _panel(
            "Agent Runs — Total by Status (24h)",
            "barchart",
            {"h": 8, "w": 12, "x": 12, "y": 0},
            [
                _prom_target(
                    "sum by (agent_name, status) (increase(agent_runs_total[24h]))",
                    "{{agent_name}} — {{status}}",
                )
            ],
        ),
        _panel(
            "Agent Duration — P50",
            "timeseries",
            {"h": 8, "w": 12, "x": 0, "y": 6},
            [
                _prom_target(
                    "histogram_quantile(0.50, rate(agent_duration_seconds_bucket[5m]))",
                    "{{agent_name}} P50",
                )
            ],
        ),
        _panel(
            "Agent Duration — P95",
            "timeseries",
            {"h": 8, "w": 12, "x": 12, "y": 8},
            [
                _prom_target(
                    "histogram_quantile(0.95, rate(agent_duration_seconds_bucket[5m]))",
                    "{{agent_name}} P95",
                )
            ],
        ),
        _panel(
            "Agent Error Rate",
            "timeseries",
            {"h": 8, "w": 24, "x": 0, "y": 16},
            [
                _prom_target(
                    'rate(agent_runs_total{status="failure"}[5m])',
                    "{{agent_name}} errors/s",
                )
            ],
        ),
    ]

    return {
        "annotations": {"list": []},
        "editable": True,
        "fiscalYearStartMonth": 0,
        "graphTooltip": 1,
        "id": None,
        "links": [],
        "panels": panels,
        "schemaVersion": 39,
        "tags": ["pipeline", "health"],
        "templating": {"list": []},
        "time": {"from": "now-6h", "to": "now"},
        "title": "Pipeline Health",
        "uid": "pipeline-health",
        "version": 1,
    }


def generate_data_quality():
    """Dashboard 2: Data Quality — anomaly count, null rate drift."""
    panels = [
        _panel(
            "Anomalies Detected — Total",
            "stat",
            {"h": 6, "w": 6, "x": 0, "y": 0},
            [_prom_target("anomalies_detected_total", "Anomalies")],
        ),
        _panel(
            "Knowledge Base Chunks",
            "stat",
            {"h": 6, "w": 6, "x": 6, "y": 0},
            [_prom_target("knowledge_base_chunks_total", "Chunks")],
        ),
        _panel(
            "Anomalies Over Time",
            "timeseries",
            {"h": 8, "w": 12, "x": 12, "y": 0},
            [_prom_target("anomalies_detected_total", "Anomalies")],
        ),
        _panel(
            "Data Drift Score by Column",
            "timeseries",
            {"h": 10, "w": 24, "x": 0, "y": 8},
            [_prom_target("data_drift_score", "{{column_name}}")],
        ),
        _panel(
            "API Request Duration",
            "heatmap",
            {"h": 8, "w": 24, "x": 0, "y": 18},
            [
                _prom_target(
                    "rate(api_request_duration_seconds_bucket[5m])", "{{endpoint}}"
                )
            ],
        ),
    ]

    return {
        "annotations": {"list": []},
        "editable": True,
        "fiscalYearStartMonth": 0,
        "graphTooltip": 1,
        "id": None,
        "links": [],
        "panels": panels,
        "schemaVersion": 39,
        "tags": ["data", "quality"],
        "templating": {"list": []},
        "time": {"from": "now-6h", "to": "now"},
        "title": "Data Quality",
        "uid": "data-quality",
        "version": 1,
    }


def generate_model_performance():
    """Dashboard 3: Model Performance — RMSE/R² by run, 3-model comparison."""
    panels = [
        _panel(
            "Best Model R² (Test)",
            "stat",
            {"h": 6, "w": 8, "x": 0, "y": 0},
            [_prom_target('model_r2{split="test"}', "{{model_name}}")],
        ),
        _panel(
            "Best Model RMSE (Test)",
            "stat",
            {"h": 6, "w": 8, "x": 8, "y": 0},
            [_prom_target('model_rmse{split="test"}', "{{model_name}}")],
        ),
        _panel(
            "Best Model MAE (Test)",
            "stat",
            {"h": 6, "w": 8, "x": 16, "y": 0},
            [_prom_target('model_mae{split="test"}', "{{model_name}}")],
        ),
        _panel(
            "RMSE by Model — Validation",
            "barchart",
            {"h": 8, "w": 12, "x": 0, "y": 6},
            [_prom_target('model_rmse{split="val"}', "{{model_name}}")],
        ),
        _panel(
            "R² by Model — Test",
            "barchart",
            {"h": 8, "w": 12, "x": 12, "y": 6},
            [_prom_target('model_r2{split="test"}', "{{model_name}}")],
        ),
        _panel(
            "RAG Query Duration",
            "timeseries",
            {"h": 8, "w": 24, "x": 0, "y": 14},
            [
                _prom_target(
                    "histogram_quantile(0.95, rate(rag_query_duration_seconds_bucket[5m]))",
                    "RAG P95",
                )
            ],
        ),
    ]

    return {
        "annotations": {"list": []},
        "editable": True,
        "fiscalYearStartMonth": 0,
        "graphTooltip": 1,
        "id": None,
        "links": [],
        "panels": panels,
        "schemaVersion": 39,
        "tags": ["model", "performance"],
        "templating": {"list": []},
        "time": {"from": "now-6h", "to": "now"},
        "title": "Model Performance",
        "uid": "model-performance",
        "version": 1,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dashboards = {
        "pipeline-health.json": generate_pipeline_health(),
        "data-quality.json": generate_data_quality(),
        "model-performance.json": generate_model_performance(),
    }

    # Wrap each dashboard in provisioning format
    for filename, dashboard in dashboards.items():
        provisioned = {
            "apiVersion": 1,
            "providers": [
                {
                    "name": filename.replace(".json", ""),
                    "orgId": 1,
                    "folder": "",
                    "type": "file",
                    "disableDeletion": False,
                    "editable": True,
                    "options": {
                        "path": "/etc/grafana/provisioning/dashboards",
                        "foldersFromFilesStructure": False,
                    },
                }
            ],
        }

        # Write the dashboard JSON
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, "w") as f:
            json.dump(dashboard, f, indent=2)
        print(f"Generated: {path}")

    # Write the provisioning config
    prov_path = os.path.join(OUTPUT_DIR, "dashboards.yml")
    with open(prov_path, "w") as f:
        import yaml

        yaml_content = """apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    editable: true
    options:
      path: /etc/grafana/provisioning/dashboards
      foldersFromFilesStructure: false
"""
        f.write(yaml_content)
    print(f"Generated: {prov_path}")


if __name__ == "__main__":
    main()
