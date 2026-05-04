"""
Agent 5 — Encoding & Scaling Agent
Transform all features into a fully numeric, model-ready matrix.
"""

from __future__ import annotations

import json
import os
import pickle

import numpy as np
import pandas as pd
from agents.base_agent import BaseAgent
from core.schemas import AgentStatus, EncodingOutput
from sklearn.preprocessing import OneHotEncoder, RobustScaler

ORDINAL_MAPS = {
    "quality_5": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0, "NA": 0},
    "finish_3": {
        "GLQ": 6,
        "ALQ": 5,
        "BLQ": 4,
        "Rec": 3,
        "LwQ": 2,
        "Unf": 1,
        "None": 0,
        "NA": 0,
        "Fin": 3,
        "RFn": 2,
    },
    "exposure_4": {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "None": 0, "NA": 0},
    "slope_3": {"Gtl": 1, "Mod": 2, "Sev": 3},
    "functional_8": {
        "Typ": 8,
        "Min1": 7,
        "Min2": 6,
        "Mod": 5,
        "Maj1": 4,
        "Maj2": 3,
        "Sev": 2,
        "Sal": 1,
        "None": 0,
    },
    "paved_drive_3": {"Y": 3, "P": 2, "N": 1, "None": 0},
}

ORDINAL_COLUMN_MAP = {
    "Exter Qual": "quality_5",
    "Exter Cond": "quality_5",
    "Bsmt Qual": "quality_5",
    "Bsmt Cond": "quality_5",
    "Heating QC": "quality_5",
    "Kitchen Qual": "quality_5",
    "Fireplace Qu": "quality_5",
    "Garage Qual": "quality_5",
    "Garage Cond": "quality_5",
    "Pool QC": "quality_5",
    "Bsmt Exposure": "exposure_4",
    "BsmtFin Type 1": "finish_3",
    "BsmtFin Type 2": "finish_3",
    "Garage Finish": "finish_3",
    "Land Slope": "slope_3",
    "Functional": "functional_8",
    "Paved Drive": "paved_drive_3",
}

OHE_COLS = [
    "MS Zoning",
    "Lot Shape",
    "Land Contour",
    "Lot Config",
    "Bldg Type",
    "House Style",
    "Roof Style",
    "Foundation",
    "Heating",
    "Central Air",
    "Sale Type",
    "Sale Condition",
]

TARGET_ENCODE_COLS = ["Neighborhood", "Exterior 1st", "Exterior 2nd"]

LOG_TRANSFORM_COLS = ["SalePrice", "Gr Liv Area", "Lot Area", "TotalSF", "1st Flr SF"]

DROP_COLS = ["Order", "PID"]


class SmoothedTargetEncoder:
    """Smoothed target encoding fitted on training fold only."""

    def __init__(self, smoothing_factor=10.0, min_samples_leaf=5):
        self.smoothing_factor = smoothing_factor
        self.min_samples_leaf = min_samples_leaf
        self.encodings_ = {}
        self.global_mean_ = 0.0

    def fit(self, X_col: pd.Series, y: pd.Series):
        self.global_mean_ = y.mean()
        stats = (
            pd.DataFrame({"val": X_col, "target": y})
            .groupby("val")["target"]
            .agg(["mean", "count"])
        )
        for cat, row in stats.iterrows():
            n = row["count"]
            smooth = 1.0 / (
                1.0 + np.exp(-(n - self.min_samples_leaf) / self.smoothing_factor)
            )
            self.encodings_[cat] = (
                smooth * row["mean"] + (1 - smooth) * self.global_mean_
            )
        return self

    def transform(self, X_col: pd.Series) -> pd.Series:
        return X_col.map(self.encodings_).fillna(self.global_mean_)


class EncodingAgent(BaseAgent):
    name = "encoding_agent"
    version = "1.0.0"

    async def execute(self, input_data) -> EncodingOutput:
        df: pd.DataFrame = self._get_df(input_data).copy()

        await self.emit(
            AgentStatus.PROGRESS,
            f"Encoding {len(df.columns)} features into model-ready matrix",
        )

        # Preserve PID and Neighborhood (original) as metadata for downstream agents
        # (anomaly_agent needs real identifiers, not encoded values)
        self._metadata_cols = {}
        if "PID" in df.columns:
            self._metadata_cols["PID"] = df["PID"].copy()
        if "Neighborhood" in df.columns:
            self._metadata_cols["Neighborhood_original"] = df["Neighborhood"].copy()

        # Drop non-feature columns
        for col in DROP_COLS:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Temporal split indices
        train_mask = df["Yr Sold"].isin([2006, 2007, 2008])
        artifacts_saved = []
        run_id = self.run_id
        base_artifacts = os.getenv("ARTIFACTS_DIR", "/app/artifacts")
        artifact_dir = f"{base_artifacts}/encoders/{run_id}"
        os.makedirs(artifact_dir, exist_ok=True)

        # 1. Ordinal encoding
        ordinal_count = 0
        for col, map_name in ORDINAL_COLUMN_MAP.items():
            if col in df.columns:
                mapping = ORDINAL_MAPS[map_name]
                df[col] = df[col].map(mapping).fillna(0).astype(int)
                ordinal_count += 1

        with open(os.path.join(artifact_dir, "ordinal_maps.json"), "w") as f:
            json.dump(ORDINAL_MAPS, f)
        artifacts_saved.append("ordinal_maps.json")
        await self.emit(
            AgentStatus.PROGRESS,
            f"Ordinal encoding: {ordinal_count} quality columns mapped ✓",
        )

        # 2. Target encoding (fitted on training fold only)
        target_encoders = {}
        te_count = 0
        for col in TARGET_ENCODE_COLS:
            if col in df.columns:
                encoder = SmoothedTargetEncoder()
                encoder.fit(df.loc[train_mask, col], df.loc[train_mask, "SalePrice"])
                df[col] = encoder.transform(df[col])
                target_encoders[col] = encoder
                te_count += 1

        with open(os.path.join(artifact_dir, "target_encoder.pkl"), "wb") as f:
            pickle.dump(target_encoders, f)
        artifacts_saved.append("target_encoder.pkl")
        await self.emit(
            AgentStatus.PROGRESS,
            f"Target encoding: {te_count} high-cardinality columns — fitted on training fold ✓",
        )

        # 3. One-hot encoding
        ohe_cols_present = [c for c in OHE_COLS if c in df.columns]
        ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        if ohe_cols_present:
            ohe.fit(df.loc[train_mask, ohe_cols_present])
            ohe_data = ohe.transform(df[ohe_cols_present])
            ohe_feature_names = ohe.get_feature_names_out(ohe_cols_present)
            ohe_df = pd.DataFrame(ohe_data, columns=ohe_feature_names, index=df.index)
            df = df.drop(columns=ohe_cols_present)
            df = pd.concat([df, ohe_df], axis=1)

        with open(os.path.join(artifact_dir, "ohe_encoder.pkl"), "wb") as f:
            pickle.dump(ohe, f)
        artifacts_saved.append("ohe_encoder.pkl")
        ohe_features = len(ohe_feature_names) if ohe_cols_present else 0
        await self.emit(
            AgentStatus.PROGRESS,
            f"One-hot encoding: {len(ohe_cols_present)} nominal columns → {ohe_features} binary features ✓",
        )

        # 4. Drop remaining string columns
        remaining_str = df.select_dtypes(include="object").columns.tolist()
        if remaining_str:
            df = df.drop(columns=remaining_str)

        # 5. Log-transform
        log_cols_applied = []
        for col in LOG_TRANSFORM_COLS:
            if col in df.columns:
                df[f"log_{col}"] = np.log1p(df[col])
                log_cols_applied.append(col)

        with open(os.path.join(artifact_dir, "log_transform_cols.json"), "w") as f:
            json.dump(log_cols_applied, f)
        artifacts_saved.append("log_transform_cols.json")
        await self.emit(
            AgentStatus.PROGRESS, f"Log-transforming: {', '.join(log_cols_applied)} ✓"
        )

        # 6. RobustScaler on continuous features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_from_scaling = ["SalePrice", "log_SalePrice", "Yr Sold"]
        scale_cols = [
            c
            for c in numeric_cols
            if c not in exclude_from_scaling and not c.startswith("log_")
        ]

        scaler = RobustScaler()
        if scale_cols:
            scaler.fit(df.loc[train_mask, scale_cols])
            df[scale_cols] = scaler.transform(df[scale_cols])

        with open(os.path.join(artifact_dir, "robust_scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
        artifacts_saved.append("robust_scaler.pkl")

        # Save feature names
        feature_names = list(df.columns)
        with open(os.path.join(artifact_dir, "feature_names_ordered.json"), "w") as f:
            json.dump(feature_names, f)
        artifacts_saved.append("feature_names_ordered.json")

        await self.emit(
            AgentStatus.PROGRESS,
            f"RobustScaler: fitted on {len(scale_cols)} continuous features ✓",
        )
        await self.emit(
            AgentStatus.PROGRESS, f"Encoder artifacts serialised to {artifact_dir} ✓"
        )

        self._df = df
        self._train_mask = train_mask

        return EncodingOutput(
            final_shape=[len(df), len(df.columns)],
            ordinal_cols_encoded=ordinal_count,
            target_encoded_cols=te_count,
            ohe_cols_encoded=len(ohe_cols_present),
            ohe_features_created=ohe_features,
            log_transformed_cols=log_cols_applied,
            scaled_cols=len(scale_cols),
            artifacts_saved=artifacts_saved,
            row_count=len(df),
        )

    def _get_df(self, input_data) -> pd.DataFrame:
        if isinstance(input_data, dict):
            for key in ["feature_agent", "cleaning_agent", "ingestion_agent"]:
                agent = input_data.get(key)
                if agent and hasattr(agent, "_df"):
                    return agent._df
        raise ValueError("No DataFrame from upstream agents")
