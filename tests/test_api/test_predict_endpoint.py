"""Tests for Predict API endpoints."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "pipeline"))
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_predict_requires_auth():
    resp = client.post("/api/predict", json={"overall_qual": 7, "gr_liv_area": 1500, "year_built": 2000, "total_bathrooms": 2, "neighborhood": "NAmes"})
    assert resp.status_code == 403


def test_predict_with_auth():
    resp = client.post(
        "/api/predict",
        headers={"X-API-Key": "changeme"},
        json={"overall_qual": 7, "gr_liv_area": 1500, "year_built": 2000, "total_bathrooms": 2, "neighborhood": "NAmes"},
    )
    assert resp.status_code == 200
    assert "predicted_price" in resp.json()


def test_batch_predict():
    resp = client.post(
        "/api/predict/batch",
        headers={"X-API-Key": "changeme"},
        json={"properties": [
            {"overall_qual": 5, "gr_liv_area": 1200, "year_built": 1990, "total_bathrooms": 1, "neighborhood": "NAmes"},
            {"overall_qual": 8, "gr_liv_area": 2000, "year_built": 2005, "total_bathrooms": 3, "neighborhood": "NridgHt"},
        ]},
    )
    assert resp.status_code == 200
    assert len(resp.json()["predictions"]) == 2
