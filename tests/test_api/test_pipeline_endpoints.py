"""Tests for Pipeline API endpoints."""

import os
import sys
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "pipeline"))
from api.main import app

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_metrics_endpoint():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"agent_duration_seconds" in resp.content


def test_run_pipeline_requires_auth():
    resp = client.post("/api/run-pipeline")
    assert resp.status_code == 403


def test_run_pipeline_with_auth():
    resp = client.post("/api/run-pipeline", headers={"X-API-Key": "changeme"})
    assert resp.status_code == 200
    assert "run_id" in resp.json()


def test_get_status_unknown_run():
    resp = client.get("/api/status/nonexistent")
    assert resp.status_code == 200
    assert resp.json()["status"] == "UNKNOWN"


def test_pipeline_runs_list():
    resp = client.get("/api/pipeline-runs")
    assert resp.status_code == 200
    assert "runs" in resp.json()
