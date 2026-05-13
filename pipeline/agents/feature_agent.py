"""
Agent 4 — Feature Engineering Agent
Create 12 new features encoding domain knowledge about residential property value.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from agents.base_agent import BaseAgent
from core.feature_engineering import FEATURE_DEFINITIONS, engineer_features
from core.schemas import AgentStatus, FeatureManifestEntry, FeatureOutput


class FeatureAgent(BaseAgent):
    name = "feature_agent"
    version = "1.0.0"

    async def execute(self, input_data: Dict[str, Any], df: pd.DataFrame = None) -> tuple[FeatureOutput, pd.DataFrame]:
        if df is None:
            raise ValueError("No DataFrame provided to FeatureAgent")

        await self.emit(
            AgentStatus.PROGRESS, "Engineering 12 features from domain knowledge"
        )

        # Use shared feature engineering module (single source of truth)
        df = engineer_features(df)

        manifest = []

        # Log feature engineering results
        r_total = df["TotalSF"].corr(df["SalePrice"]) if "SalePrice" in df.columns else 0
        r_porch = df["PorchSF"].corr(df["SalePrice"]) if "SalePrice" in df.columns else 0
        await self.emit(
            AgentStatus.PROGRESS, f"Area composites: TotalSF (r={r_total:.3f}) ✓"
        )

        await self.emit(
            AgentStatus.PROGRESS,
            "Age features: HouseAge, RemodAge, GarageAge — leakage-safe ✓",
        )

        await self.emit(AgentStatus.PROGRESS, "Bathroom composite: TotalBathrooms ✓")

        await self.emit(
            AgentStatus.PROGRESS,
            "Presence flags: HasPool, HasGarage, HasBasement, HasFireplace, IsNew ✓",
        )
        
        await self.emit(AgentStatus.PROGRESS, "Interaction: OverallScore ✓")

        # Build manifest with correlations
        engineered_cols = [
            "TotalSF", "PorchSF", "HouseAge", "RemodAge", "GarageAge",
            "TotalBathrooms", "HasPool", "HasGarage", "HasBasement",
            "HasFireplace", "IsNew", "OverallScore",
        ]

        for i, (name, formula, rationale) in enumerate(FEATURE_DEFINITIONS):
            corr = float(df[name].corr(df["SalePrice"])) if name in df.columns and "SalePrice" in df.columns else 0.0
            manifest.append(
                FeatureManifestEntry(
                    name=name,
                    formula=formula,
                    rationale=rationale,
                    pearson_correlation=round(corr, 4),
                )
            )

        # Post-engineering assertions
        assert (
            df[engineered_cols].isnull().sum().sum() == 0
        ), "Engineered features contain nulls"
        
        if "TotalSF" in df.columns and "1st Flr SF" in df.columns:
            assert (df["TotalSF"] >= df["1st Flr SF"]).all(), "TotalSF < 1stFlrSF violation"
        elif "TotalSF" in df.columns:
            assert (df["TotalSF"] >= 0).all(), "Negative TotalSF found"
        
        if "HouseAge" in df.columns:
            assert (df["HouseAge"] >= 0).all(), "Negative HouseAge found"

        await self.emit(
            AgentStatus.PROGRESS,
            f"Post-engineering assertions: 0 nulls, all values valid ✓",
            progress_pct=100.0,
        )

        output = FeatureOutput(
            features_created=12,
            feature_manifest=manifest,
            total_columns=len(df.columns),
            row_count=len(df),
        )
        return output, df
