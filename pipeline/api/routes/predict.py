"""
Prediction API Routes — single and batch prediction endpoints.
"""
from __future__ import annotations
import json
import os
import pickle
import numpy as np
from fastapi import APIRouter, Depends
from core.schemas import PredictRequest, PredictResponse, BatchPredictRequest, BatchPredictResponse
from api.middleware import verify_api_key

router = APIRouter()


def _load_model():
    """Load the latest best model and feature columns."""
    artifacts_dir = "/app/artifacts/models"
    if not os.path.exists(artifacts_dir):
        return None, None
    runs = sorted(os.listdir(artifacts_dir), reverse=True)
    for run_id in runs:
        model_path = os.path.join(artifacts_dir, run_id, "best_model.pkl")
        cols_path = os.path.join(artifacts_dir, run_id, "feature_cols.json")
        if os.path.exists(model_path) and os.path.exists(cols_path):
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            with open(cols_path, "r") as f:
                feature_cols = json.load(f)
            return model, feature_cols
    return None, None


def _prepare_features(req: PredictRequest, feature_cols: list) -> np.ndarray:
    """Build feature vector from prediction request."""
    features = {col: 0.0 for col in feature_cols}
    mapping = {
        "Overall Qual": req.overall_qual,
        "Gr Liv Area": req.gr_liv_area,
        "Year Built": req.year_built,
        "TotalBathrooms": req.total_bathrooms,
        "Garage Area": req.garage_area,
        "Total Bsmt SF": req.total_bsmt_sf,
        "1st Flr SF": req.first_flr_sf,
        "Full Bath": req.full_bath,
    }
    for col, val in mapping.items():
        if col in features:
            features[col] = float(val)

    return np.array([features[col] for col in feature_cols]).reshape(1, -1)


@router.post("/predict", response_model=PredictResponse)
async def predict_single(req: PredictRequest, api_key: str = Depends(verify_api_key)):
    model, feature_cols = _load_model()
    if model is None:
        return PredictResponse(
            predicted_price=0.0,
            shap_top_features={"error_no_model": -1.0},
        )

    X = _prepare_features(req, feature_cols)
    pred_log = model.predict(X)[0]
    pred_price = float(np.expm1(pred_log)) if pred_log < 20 else float(pred_log)

    # SHAP top features (simplified)
    shap_features = {}
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)[0]
        top_indices = np.argsort(np.abs(shap_values))[-5:][::-1]
        for idx in top_indices:
            shap_features[feature_cols[idx]] = round(float(shap_values[idx]), 4)
    except Exception:
        shap_features = {"Overall Qual": 0.0}

    return PredictResponse(
        predicted_price=round(pred_price, 2),
        confidence_interval=[round(pred_price * 0.9, 2), round(pred_price * 1.1, 2)],
        shap_top_features=shap_features,
    )


@router.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(req: BatchPredictRequest, api_key: str = Depends(verify_api_key)):
    model, feature_cols = _load_model()
    predictions = []
    for prop in req.properties:
        if model is None:
            predictions.append(PredictResponse(predicted_price=0.0))
            continue
        X = _prepare_features(prop, feature_cols)
        pred_log = model.predict(X)[0]
        pred_price = float(np.expm1(pred_log)) if pred_log < 20 else float(pred_log)
        predictions.append(PredictResponse(
            predicted_price=round(pred_price, 2),
            confidence_interval=[round(pred_price * 0.9, 2), round(pred_price * 1.1, 2)],
        ))
    return BatchPredictResponse(predictions=predictions)
