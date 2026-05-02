"""
Agent 4 — Feature Engineering Agent
Create 12 new features encoding domain knowledge about residential property value.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from agents.base_agent import BaseAgent
from core.schemas import AgentStatus, FeatureManifestEntry, FeatureOutput

FEATURE_DEFINITIONS = [
    ("TotalSF", "TotalBsmtSF + 1stFlrSF + 2ndFlrSF", "Combined livable space is primary value driver"),
    ("PorchSF", "WoodDeckSF + OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch", "Outdoor usable space adds marginal value"),
    ("HouseAge", "YrSold - YearBuilt", "Relative age prevents temporal leakage"),
    ("RemodAge", "YrSold - YearRemod/Add", "Recency of remodel affects value"),
    ("GarageAge", "YrSold - GarageYrBlt", "Garage condition proxy"),
    ("TotalBathrooms", "FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath", "Half-bath = 0.5 full-bath equivalent"),
    ("HasPool", "PoolArea > 0", "Presence more predictive than raw area"),
    ("HasGarage", "GarageArea > 0", "Binary garage presence"),
    ("HasBasement", "TotalBsmtSF > 0", "Binary basement presence"),
    ("HasFireplace", "Fireplaces > 0", "Binary fireplace presence"),
    ("IsNew", "YearBuilt == YrSold", "New construction premium"),
    ("OverallScore", "OverallQual * OverallCond", "Quality × condition interaction"),
]


class FeatureAgent(BaseAgent):
    name = "feature_agent"
    version = "1.0.0"

    async def execute(self, input_data) -> FeatureOutput:
        df: pd.DataFrame = self._get_df(input_data).copy()

        await self.emit(AgentStatus.PROGRESS, "Engineering 12 features from domain knowledge")

        manifest = []

        # Area composites
        df["TotalSF"] = df["Total Bsmt SF"] + df["1st Flr SF"] + df["2nd Flr SF"]
        df["PorchSF"] = (df["Wood Deck SF"] + df["Open Porch SF"] +
                         df["Enclosed Porch"] + df["3Ssn Porch"] + df["Screen Porch"])
        r_total = df["TotalSF"].corr(df["SalePrice"])
        r_porch = df["PorchSF"].corr(df["SalePrice"])
        await self.emit(AgentStatus.PROGRESS, f"Area composites: TotalSF (r={r_total:.3f}) ✓")

        # Age features (relative to sale year)
        df["HouseAge"] = df["Yr Sold"] - df["Year Built"]
        df["RemodAge"] = df["Yr Sold"] - df["Year Remod/Add"]
        df["GarageAge"] = df["Yr Sold"] - df["Garage Yr Blt"]
        df["GarageAge"] = df["GarageAge"].fillna(df["HouseAge"])
        await self.emit(AgentStatus.PROGRESS, "Age features: HouseAge, RemodAge, GarageAge — leakage-safe ✓")

        # Bathroom composite
        df["TotalBathrooms"] = (df["Full Bath"] + 0.5 * df["Half Bath"] +
                                df["Bsmt Full Bath"] + 0.5 * df["Bsmt Half Bath"])
        await self.emit(AgentStatus.PROGRESS, "Bathroom composite: TotalBathrooms ✓")

        # Binary presence flags
        df["HasPool"] = (df["Pool Area"] > 0).astype(int)
        df["HasGarage"] = (df["Garage Area"] > 0).astype(int)
        df["HasBasement"] = (df["Total Bsmt SF"] > 0).astype(int)
        df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
        df["IsNew"] = (df["Year Built"] == df["Yr Sold"]).astype(int)
        await self.emit(AgentStatus.PROGRESS, "Presence flags: HasPool, HasGarage, HasBasement, HasFireplace, IsNew ✓")

        # Interaction feature
        df["OverallScore"] = df["Overall Qual"] * df["Overall Cond"]
        await self.emit(AgentStatus.PROGRESS, "Interaction: OverallScore ✓")

        # Build manifest with correlations
        engineered_cols = [
            "TotalSF", "PorchSF", "HouseAge", "RemodAge", "GarageAge",
            "TotalBathrooms", "HasPool", "HasGarage", "HasBasement",
            "HasFireplace", "IsNew", "OverallScore",
        ]

        for i, (name, formula, rationale) in enumerate(FEATURE_DEFINITIONS):
            corr = float(df[name].corr(df["SalePrice"])) if name in df.columns else 0.0
            manifest.append(FeatureManifestEntry(
                name=name, formula=formula, rationale=rationale,
                pearson_correlation=round(corr, 4),
            ))

        # Post-engineering assertions
        assert df[engineered_cols].isnull().sum().sum() == 0, "Engineered features contain nulls"
        assert (df["TotalSF"] >= df["1st Flr SF"]).all(), "TotalSF < 1stFlrSF violation"
        assert (df["HouseAge"] >= 0).all(), "Negative HouseAge found"

        await self.emit(
            AgentStatus.PROGRESS,
            f"Post-engineering assertions: 0 nulls, all values valid ✓",
            progress_pct=100.0,
        )

        self._df = df

        return FeatureOutput(
            features_created=12,
            feature_manifest=manifest,
            total_columns=len(df.columns),
            row_count=len(df),
        )

    def _get_df(self, input_data) -> pd.DataFrame:
        if isinstance(input_data, dict):
            for key in ["cleaning_agent", "schema_agent", "ingestion_agent"]:
                agent = input_data.get(key)
                if agent and hasattr(agent, "_df"):
                    return agent._df
        raise ValueError("No DataFrame from upstream agents")
