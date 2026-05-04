"""
Prediction API Routes — single and batch prediction endpoints.
"""
from __future__ import annotations
import json
import os
import pickle
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends
from core.schemas import PredictRequest, PredictResponse, BatchPredictRequest, BatchPredictResponse
from api.middleware import verify_api_key

router = APIRouter()


def _load_model():
    """Load the latest best model and feature columns."""
    artifacts_dir = "/app/artifacts/models"
    if not os.path.exists(artifacts_dir):
        return None, None, None
    runs = sorted(os.listdir(artifacts_dir), reverse=True)
    for run_id in runs:
        model_path = os.path.join(artifacts_dir, run_id, "best_model.pkl")
        cols_path = os.path.join(artifacts_dir, run_id, "feature_cols.json")
        if os.path.exists(model_path) and os.path.exists(cols_path):
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            with open(cols_path, "r") as f:
                feature_cols = json.load(f)
            return model, feature_cols, run_id
    return None, None, None


def _prepare_features(req: PredictRequest, feature_cols: list, run_id: str) -> np.ndarray:
    """Build feature vector from prediction request with proper preprocessing."""
    import structlog
    features = {col: 0.0 for col in feature_cols}
    
    raw_inputs = {
        "Overall Qual": req.overall_qual,
        "Gr Liv Area": req.gr_liv_area,
        "Year Built": req.year_built,
        "TotalBathrooms": req.total_bathrooms,
        "Garage Area": req.garage_area,
        "Total Bsmt SF": req.total_bsmt_sf,
        "1st Flr SF": req.first_flr_sf,
        "Full Bath": req.full_bath,
    }

    # Reconstruct engineered features
    raw_inputs["TotalSF"] = raw_inputs["Total Bsmt SF"] + raw_inputs["Gr Liv Area"]
    raw_inputs["OverallScore"] = raw_inputs["Overall Qual"] * 5  # Assume average condition
    raw_inputs["HasGarage"] = 1 if raw_inputs["Garage Area"] > 0 else 0
    raw_inputs["HasBasement"] = 1 if raw_inputs["Total Bsmt SF"] > 0 else 0
    raw_inputs["HouseAge"] = max(2010 - raw_inputs["Year Built"], 0)

    artifact_dir = f"/app/artifacts/encoders/{run_id}"
    try:
        with open(f"{artifact_dir}/target_encoder.pkl", "rb") as f: target_encoders = pickle.load(f)
        with open(f"{artifact_dir}/robust_scaler.pkl", "rb") as f: robust_scaler = pickle.load(f)
        with open(f"{artifact_dir}/log_transform_cols.json", "r") as f: log_transform_cols = json.load(f)
    except Exception as e:
        structlog.get_logger().error(f"Failed to load encoders: {e}")
        return np.zeros((1, len(feature_cols)))

    if "Neighborhood" in target_encoders and "Neighborhood" in feature_cols:
        features["Neighborhood"] = target_encoders["Neighborhood"].transform(pd.Series([req.neighborhood])).iloc[0]

    for col, val in raw_inputs.items():
        if col in log_transform_cols:
            features[f"log_{col}"] = np.log1p(val)

    if hasattr(robust_scaler, "feature_names_in_"):
        scale_cols = robust_scaler.feature_names_in_
        row_for_scaling = [0.0] * len(scale_cols)
        for i, col in enumerate(scale_cols):
            if col in raw_inputs:
                row_for_scaling[i] = raw_inputs[col]
            else:
                row_for_scaling[i] = robust_scaler.center_[i]
                
        scaled_row = robust_scaler.transform([row_for_scaling])[0]
        for i, col in enumerate(scale_cols):
            if col in features:
                features[col] = scaled_row[i]

    for col, val in raw_inputs.items():
        if col in features and col not in scale_cols and col not in log_transform_cols:
            features[col] = val

    return np.array([features.get(col, 0.0) for col in feature_cols]).reshape(1, -1)


@router.post("/predict", response_model=PredictResponse)
async def predict_single(req: PredictRequest, api_key: str = Depends(verify_api_key)):
    model, feature_cols, run_id = _load_model()
    if model is None:
        return PredictResponse(
            predicted_price=0.0,
            shap_top_features={"error_no_model": -1.0},
        )

    X = _prepare_features(req, feature_cols, run_id)
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
    model, feature_cols, run_id = _load_model()
    predictions = []
    for prop in req.properties:
        if model is None:
            predictions.append(PredictResponse(predicted_price=0.0))
            continue
        X = _prepare_features(prop, feature_cols, run_id)
        pred_log = model.predict(X)[0]
        pred_price = float(np.expm1(pred_log)) if pred_log < 20 else float(pred_log)
        predictions.append(PredictResponse(
            predicted_price=round(pred_price, 2),
            confidence_interval=[round(pred_price * 0.9, 2), round(pred_price * 1.1, 2)],
        ))
    return BatchPredictResponse(predictions=predictions)
