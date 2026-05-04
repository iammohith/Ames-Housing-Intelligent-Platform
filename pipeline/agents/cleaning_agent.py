"""
Agent 3 — Data Cleaning Agent
Fix every known data quality issue with explicit, documented, auditable logic.
"""

from __future__ import annotations

import os

import pandas as pd
from agents.base_agent import BaseAgent
from core.schemas import (AgentStatus, ArtifactFlag, CleaningOutput,
                          CleaningReport, ImputationRecord)

STRING_STRUCTURAL_NA = [
    "Alley",
    "Pool QC",
    "Misc Feature",
    "Fence",
    "Fireplace Qu",
    "Garage Type",
    "Garage Finish",
    "Garage Qual",
    "Garage Cond",
    "Bsmt Qual",
    "Bsmt Cond",
    "Bsmt Exposure",
    "BsmtFin Type 1",
    "BsmtFin Type 2",
]


class CleaningAgent(BaseAgent):
    name = "cleaning_agent"
    version = "1.0.0"

    async def execute(self, input_data) -> CleaningOutput:
        df: pd.DataFrame = self._get_df(input_data).copy()
        rows_in = len(df)

        await self.emit(
            AgentStatus.PROGRESS, f"Beginning data cleaning: {rows_in:,} rows"
        )

        structural_na_fills = {}
        imputed_cols = {}

        # Step 1 — Structural NA resolution (14 columns)
        for col in STRING_STRUCTURAL_NA:
            if col in df.columns:
                count = df[col].isnull().sum()
                if count > 0:
                    df[col] = df[col].fillna("None")
                    structural_na_fills[col] = int(count)

        total_struct_fills = sum(structural_na_fills.values())
        await self.emit(
            AgentStatus.PROGRESS,
            f"Step 1/6: Structural NA resolution ({len(structural_na_fills)} columns) — "
            f"{total_struct_fills:,} cells filled",
            progress_pct=16.0,
        )

        # Step 2 — Statistical imputation
        # LotFrontage: neighborhood median
        if "Lot Frontage" in df.columns:
            lf_nulls = df["Lot Frontage"].isnull().sum()
            df["Lot Frontage"] = df.groupby("Neighborhood")["Lot Frontage"].transform(
                lambda x: x.fillna(x.median())
            )
            # Fill any remaining with global median
            df["Lot Frontage"] = df["Lot Frontage"].fillna(df["Lot Frontage"].median())
            imputed_cols["Lot Frontage"] = ImputationRecord(
                column="Lot Frontage",
                method="neighborhood_median",
                rows_affected=int(lf_nulls),
            )

        # GarageYrBlt: fill with YearBuilt
        if "Garage Yr Blt" in df.columns and "Year Built" in df.columns:
            gyb_nulls = df["Garage Yr Blt"].isnull().sum()
            df["Garage Yr Blt"] = df["Garage Yr Blt"].fillna(df["Year Built"])
            imputed_cols["Garage Yr Blt"] = ImputationRecord(
                column="Garage Yr Blt",
                method="fill_with_YearBuilt",
                rows_affected=int(gyb_nulls),
            )

        # MasVnrType: no masonry = "None"
        if "Mas Vnr Type" in df.columns:
            mvt_nulls = df["Mas Vnr Type"].isnull().sum()
            df["Mas Vnr Type"] = df["Mas Vnr Type"].fillna("None")
            imputed_cols["Mas Vnr Type"] = ImputationRecord(
                column="Mas Vnr Type",
                method="fill_None",
                rows_affected=int(mvt_nulls),
            )

        # MasVnrArea: no masonry = 0
        if "Mas Vnr Area" in df.columns:
            mva_nulls = df["Mas Vnr Area"].isnull().sum()
            df["Mas Vnr Area"] = df["Mas Vnr Area"].fillna(0)
            imputed_cols["Mas Vnr Area"] = ImputationRecord(
                column="Mas Vnr Area",
                method="fill_zero",
                rows_affected=int(mva_nulls),
            )

        imputed_total = sum(r.rows_affected for r in imputed_cols.values())
        await self.emit(
            AgentStatus.PROGRESS,
            f"Step 2/6: Statistical imputation — {imputed_total} rows imputed across "
            f"{len(imputed_cols)} columns",
            progress_pct=33.0,
        )

        # Step 3 — Row-level drops (Electrical null)
        rows_before_drop = len(df)
        if "Electrical" in df.columns:
            df = df.dropna(subset=["Electrical"])
        rows_dropped = rows_before_drop - len(df)

        await self.emit(
            AgentStatus.PROGRESS,
            f"Step 3/6: Dropped {rows_dropped} row(s) (Electrical null)",
            progress_pct=50.0,
        )

        # Step 4 — String standardization
        str_cols = df.select_dtypes(include=["object", "string"]).columns
        string_fixes = 0
        for col in str_cols:
            original = df[col].copy()
            df[col] = df[col].str.strip()
            changes = (original != df[col]).sum()
            string_fixes += int(changes)

        await self.emit(
            AgentStatus.PROGRESS,
            f"Step 4/6: String standardization — {string_fixes} cells normalised",
            progress_pct=70.0,
        )

        # Step 5 — Artifact detection
        artifact_flags = []
        if "Gr Liv Area" in df.columns and "SalePrice" in df.columns:
            artifacts = df[(df["Gr Liv Area"] > 4000) & (df["SalePrice"] < 200000)]
            for idx, row in artifacts.iterrows():
                pid_val = str(row.get("PID", idx))
                artifact_flags.append(
                    ArtifactFlag(
                        index=int(idx),
                        pid=pid_val,
                        reason="GrLivArea>4000 & SalePrice<200k",
                        gr_liv_area=float(row["Gr Liv Area"]),
                        sale_price=float(row["SalePrice"]),
                    )
                )
                df.at[idx, "artifact_flag"] = True

            if (
                os.getenv("REMOVE_ARTIFACTS", "true").lower() == "true"
                and len(artifacts) > 0
            ):
                df = df[df.get("artifact_flag") != True].copy()
                if "artifact_flag" in df.columns:
                    df = df.drop(columns=["artifact_flag"])

        await self.emit(
            AgentStatus.PROGRESS,
            f"Step 5/6: Artifact detection — {len(artifact_flags)} flagged",
            progress_pct=85.0,
        )

        # Step 6 — Catch-all: fill any remaining nulls
        # Numeric columns: fill with 0 (e.g. GarageCars, BsmtFullBath when no garage/basement)
        # String columns: fill with "None"
        remaining_before = int(df.isnull().sum().sum())
        num_cols = df.select_dtypes(include=["number"]).columns
        for col in num_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(0)
        cat_cols = df.select_dtypes(include=["object", "string"]).columns
        for col in cat_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna("None")

        remaining_after = int(df.isnull().sum().sum())
        catchall_fills = remaining_before - remaining_after
        if catchall_fills > 0:
            await self.emit(
                AgentStatus.PROGRESS,
                f"Step 6/6: Catch-all imputation — {catchall_fills} remaining nulls filled",
                progress_pct=95.0,
            )

        # Post-condition assertions
        remaining_nulls = int(df.isnull().sum().sum())
        assert remaining_nulls == 0, f"BUG: Cleaning agent left {remaining_nulls} nulls"
        assert (df["SalePrice"] > 0).all(), "BUG: Non-positive SalePrice found"

        rows_out = len(df)
        await self.emit(
            AgentStatus.PROGRESS,
            f"Post-clean assertion: NULL rate = 0.000 ✓",
            progress_pct=100.0,
        )

        # Store cleaned DataFrame
        self._df = df

        return CleaningOutput(
            cleaning_report=CleaningReport(
                rows_in=rows_in,
                rows_out=rows_out,
                rows_dropped=rows_in - rows_out,
                structural_na_fills=structural_na_fills,
                imputed_cols=imputed_cols,
                string_fixes=string_fixes,
                artifact_flags=artifact_flags,
                post_clean_null_rate=0.0,
            ),
            row_count=rows_out,
        )

    def _get_df(self, input_data) -> pd.DataFrame:
        if isinstance(input_data, dict):
            for key in ["ingestion_agent", "schema_agent"]:
                agent = input_data.get(key)
                if agent and hasattr(agent, "_df"):
                    return agent._df
            # Check result objects
            for val in input_data.values():
                if hasattr(val, "_df"):
                    return val._df
        raise ValueError("No DataFrame available from upstream agents")
