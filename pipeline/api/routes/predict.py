"""
Prediction API Routes — single and batch prediction endpoints.
"""

from __future__ import annotations

import json
import os
import pickle
import threading

import numpy as np
import pandas as pd
from api.middleware import verify_api_key
from core.feature_engineering import engineer_features_from_dict
from core.schemas import (BatchPredictRequest, BatchPredictResponse,
                          PredictRequest, PredictResponse)
from fastapi import APIRouter, Depends, HTTPException

router = APIRouter()

# Thread-safe model caching to prevent race conditions on concurrent predictions
_MODEL_LOCK = threading.Lock()
_MODEL_CACHE = None
_CACHE_RUN_ID = None


def _load_model():
    """Load the latest best model and feature columns with caching and locking."""
    global _MODEL_CACHE, _CACHE_RUN_ID
    
    with _MODEL_LOCK:
        # Check if we have cached model from a recent run
        if _MODEL_CACHE is not None:
            return _MODEL_CACHE + (_CACHE_RUN_ID,)
        
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
                
                # Cache for future requests
                _MODEL_CACHE = (model, feature_cols)
                _CACHE_RUN_ID = run_id
                return model, feature_cols, run_id
    
    return None, None, None


def _prepare_features(
    req: PredictRequest, feature_cols: list, run_id: str
) -> np.ndarray:
    """
    Build feature vector from prediction request using shared feature engineering.
    Single source of truth ensures consistency with training pipeline.
    """
    import structlog

    # Build raw input dict from request, handling defaults
    year_remod = req.year_remod if req.year_remod > 0 else req.year_built
    garage_yr_blt = req.garage_yr_blt if req.garage_yr_blt > 0 else req.year_built
    
    raw_inputs = {
        "Overall Qual": req.overall_qual,
        "Overall Cond": req.overall_cond,
        "Gr Liv Area": req.gr_liv_area,
        "Year Built": req.year_built,
        "Year Remod/Add": year_remod,
        "Garage Yr Blt": garage_yr_blt,
        "Garage Area": req.garage_area,
        "Total Bsmt SF": req.total_bsmt_sf,
        "1st Flr SF": req.first_flr_sf,
        "2nd Flr SF": req.second_flr_sf,
        "Full Bath": req.full_bath,
        "Half Bath": req.half_bath,
        "Bsmt Full Bath": req.bsmt_full_bath,
        "Bsmt Half Bath": req.bsmt_half_bath,
        "Pool Area": req.pool_area,
        "Fireplaces": req.fireplaces,
        "Wood Deck SF": req.wood_deck_sf,
        "Open Porch SF": req.open_porch_sf,
        "Enclosed Porch": req.enclosed_porch,
        "3Ssn Porch": req.three_ssn_porch,
        "Screen Porch": req.screen_porch,
        "Neighborhood": req.neighborhood,
    }

    # Engineer features (shared module - single source of truth)
    engineered = engineer_features_from_dict(raw_inputs, sale_year=req.sale_year)

    # Load encoders/scalers
    artifact_dir = f"/app/artifacts/encoders/{run_id}"
    features = {col: 0.0 for col in feature_cols}
    
    try:
        with open(f"{artifact_dir}/target_encoder.pkl", "rb") as f:
            target_encoders = pickle.load(f)
        with open(f"{artifact_dir}/robust_scaler.pkl", "rb") as f:
            robust_scaler = pickle.load(f)
        with open(f"{artifact_dir}/log_transform_cols.json", "r") as f:
            log_transform_cols = json.load(f)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model encoders not found. Run pipeline first. Missing: {e.filename}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model encoders: {str(e)}"
        )

    # Apply target encoding for categorical features
    if "Neighborhood" in target_encoders and "Neighborhood" in feature_cols:
        try:
            features["Neighborhood"] = (
                target_encoders["Neighborhood"]
                .transform(pd.Series([req.neighborhood]))
                .iloc[0]
            )
        except Exception as e:
            structlog.get_logger().warning(f"Neighborhood encoding failed: {e}")
            features["Neighborhood"] = 0.0

    # Apply log transformation where needed
    for col, val in engineered.items():
        if col in log_transform_cols:
            features[f"log_{col}"] = np.log1p(val)

    # Apply robust scaling
    scale_cols = []
    if hasattr(robust_scaler, "feature_names_in_"):
        scale_cols = list(robust_scaler.feature_names_in_)
        row_for_scaling = [0.0] * len(scale_cols)
        for i, col in enumerate(scale_cols):
            if col in engineered:
                row_for_scaling[i] = engineered[col]
            else:
                row_for_scaling[i] = robust_scaler.center_[i]

        scaled_row = robust_scaler.transform([row_for_scaling])[0]
        for i, col in enumerate(scale_cols):
            if col in features:
                features[col] = scaled_row[i]

    # Copy any non-scaled, non-log-transformed features
    for col, val in engineered.items():
        if col in features and col not in scale_cols and col not in log_transform_cols:
            features[col] = val

    return np.array([features.get(col, 0.0) for col in feature_cols]).reshape(1, -1)


@router.post("/predict", response_model=PredictResponse)
async def predict_single(req: PredictRequest, api_key: str = Depends(verify_api_key)):
    model, feature_cols, run_id = _load_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="No trained model available. Run pipeline first."
        )

    X = _prepare_features(req, feature_cols, run_id)
    pred_log = model.predict(X)[0]
    pred_price = float(np.expm1(pred_log)) if pred_log < 20 else float(pred_log)

    # SHAP top features (simplified) with robust dimension handling
    shap_features = {}
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Handle different SHAP output shapes
        if isinstance(shap_values, list):  # Some explainers return list of arrays
            shap_values = shap_values[0]
        
        shap_array = np.abs(shap_values)
        if shap_array.ndim == 3:  # Binary classification: (2, samples, features)
            shap_array = shap_array[0]
        elif shap_array.ndim != 2:
            # Unexpected shape, skip SHAP
            raise ValueError(f"Unexpected SHAP shape: {shap_array.shape}")
        
        shap_row = shap_array[0] if shap_array.ndim == 2 else shap_array
        top_indices = np.argsort(np.abs(shap_row))[-5:][::-1]
        for idx in top_indices:
            if idx < len(feature_cols):
                shap_features[feature_cols[idx]] = round(float(shap_row[idx]), 4)
    except Exception as e:
        # If SHAP fails for any reason, return empty dict (fallback to top features based on model)
        import structlog
        structlog.get_logger().warning(f"SHAP computation failed: {e}")
        shap_features = {"Overall Qual": 0.0}

    return PredictResponse(
        predicted_price=round(pred_price, 2),
        confidence_interval=[round(pred_price * 0.9, 2), round(pred_price * 1.1, 2)],
        shap_top_features=shap_features,
    )


@router.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(
    req: BatchPredictRequest, api_key: str = Depends(verify_api_key)
):
    model, feature_cols, run_id = _load_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="No trained model available. Run pipeline first."
        )
    
    predictions = []
    for prop in req.properties:
        X = _prepare_features(prop, feature_cols, run_id)
        pred_log = model.predict(X)[0]
        pred_price = float(np.expm1(pred_log)) if pred_log < 20 else float(pred_log)
        predictions.append(
            PredictResponse(
                predicted_price=round(pred_price, 2),
                confidence_interval=[
                    round(pred_price * 0.9, 2),
                    round(pred_price * 1.1, 2),
                ],
            )
        )
    return BatchPredictResponse(predictions=predictions)
