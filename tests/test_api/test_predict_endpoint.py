"""Tests for Predict API endpoints."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "pipeline"))
from api.main import app
from fastapi.testclient import TestClient
from unittest.mock import patch
import numpy as np

client = TestClient(app)

class MockModel:
    def predict(self, X):
        return np.array([12.0] * len(X))

def mock_load_model():
    return MockModel(), ["Overall Qual", "Gr Liv Area"], "mock-run-id"



def test_predict_requires_auth():
    resp = client.post(
        "/api/predict",
        json={
            "overall_qual": 7,
            "gr_liv_area": 1500,
            "year_built": 2000,
            "total_bathrooms": 2,
            "neighborhood": "NAmes",
        },
    )
    assert resp.status_code == 403


@patch("api.routes.predict._load_model", side_effect=mock_load_model)
@patch("api.routes.predict._prepare_features", return_value=np.array([[1.0, 1.0]]))
def test_predict_with_auth(mock_prep, mock_load):
    resp = client.post(
        "/api/predict",
        headers={"X-API-Key": "changeme"},
        json={
            "overall_qual": 7,
            "gr_liv_area": 1500,
            "year_built": 2000,
            "total_bathrooms": 2,
            "neighborhood": "NAmes",
        },
    )
    assert resp.status_code == 200
    assert "predicted_price" in resp.json()


@patch("api.routes.predict._load_model", side_effect=mock_load_model)
@patch("api.routes.predict._prepare_features", return_value=np.array([[1.0, 1.0]]))
def test_batch_predict(mock_prep, mock_load):
    resp = client.post(
        "/api/predict/batch",
        headers={"X-API-Key": "changeme"},
        json={
            "properties": [
                {
                    "overall_qual": 5,
                    "gr_liv_area": 1200,
                    "year_built": 1990,
                    "total_bathrooms": 1,
                    "neighborhood": "NAmes",
                },
                {
                    "overall_qual": 8,
                    "gr_liv_area": 2000,
                    "year_built": 2005,
                    "total_bathrooms": 3,
                    "neighborhood": "NridgHt",
                },
            ]
        },
    )
    assert resp.status_code == 200
    assert len(resp.json()["predictions"]) == 2
