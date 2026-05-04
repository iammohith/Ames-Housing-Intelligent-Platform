"""
Agent 2 — Schema Validation Agent
Guarantee structural integrity before any transformation.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd
from agents.base_agent import BaseAgent
from core.schemas import (AgentStatus, ColumnInfo, ColumnType, SchemaOutput,
                          SchemaReport)
from rapidfuzz import fuzz, process

# Expected 82-column contract (standardized names)
EXPECTED_COLUMNS = [
    "Order",
    "PID",
    "MS SubClass",
    "MS Zoning",
    "Lot Frontage",
    "Lot Area",
    "Street",
    "Alley",
    "Lot Shape",
    "Land Contour",
    "Utilities",
    "Lot Config",
    "Land Slope",
    "Neighborhood",
    "Condition 1",
    "Condition 2",
    "Bldg Type",
    "House Style",
    "Overall Qual",
    "Overall Cond",
    "Year Built",
    "Year Remod/Add",
    "Roof Style",
    "Roof Matl",
    "Exterior 1st",
    "Exterior 2nd",
    "Mas Vnr Type",
    "Mas Vnr Area",
    "Exter Qual",
    "Exter Cond",
    "Foundation",
    "Bsmt Qual",
    "Bsmt Cond",
    "Bsmt Exposure",
    "BsmtFin Type 1",
    "BsmtFin SF 1",
    "BsmtFin Type 2",
    "BsmtFin SF 2",
    "Bsmt Unf SF",
    "Total Bsmt SF",
    "Heating",
    "Heating QC",
    "Central Air",
    "Electrical",
    "1st Flr SF",
    "2nd Flr SF",
    "Low Qual Fin SF",
    "Gr Liv Area",
    "Bsmt Full Bath",
    "Bsmt Half Bath",
    "Full Bath",
    "Half Bath",
    "Bedroom AbvGr",
    "Kitchen AbvGr",
    "Kitchen Qual",
    "TotRms AbvGrd",
    "Functional",
    "Fireplaces",
    "Fireplace Qu",
    "Garage Type",
    "Garage Yr Blt",
    "Garage Finish",
    "Garage Cars",
    "Garage Area",
    "Garage Qual",
    "Garage Cond",
    "Paved Drive",
    "Wood Deck SF",
    "Open Porch SF",
    "Enclosed Porch",
    "3Ssn Porch",
    "Screen Porch",
    "Pool Area",
    "Pool QC",
    "Fence",
    "Misc Feature",
    "Misc Val",
    "Mo Sold",
    "Yr Sold",
    "Sale Type",
    "Sale Condition",
    "SalePrice",
]

NUMERIC_CONTINUOUS = {
    "Lot Frontage",
    "Lot Area",
    "Mas Vnr Area",
    "BsmtFin SF 1",
    "BsmtFin SF 2",
    "Bsmt Unf SF",
    "Total Bsmt SF",
    "1st Flr SF",
    "2nd Flr SF",
    "Low Qual Fin SF",
    "Gr Liv Area",
    "Garage Area",
    "Wood Deck SF",
    "Open Porch SF",
    "Enclosed Porch",
    "3Ssn Porch",
    "Screen Porch",
    "Pool Area",
    "Misc Val",
    "SalePrice",
}

NUMERIC_DISCRETE = {
    "Order",
    "MS SubClass",
    "Overall Qual",
    "Overall Cond",
    "Year Built",
    "Year Remod/Add",
    "Bsmt Full Bath",
    "Bsmt Half Bath",
    "Full Bath",
    "Half Bath",
    "Bedroom AbvGr",
    "Kitchen AbvGr",
    "TotRms AbvGrd",
    "Fireplaces",
    "Garage Yr Blt",
    "Garage Cars",
    "Mo Sold",
    "Yr Sold",
}

ORDINAL_CATEGORICAL = {
    "Exter Qual",
    "Exter Cond",
    "Bsmt Qual",
    "Bsmt Cond",
    "Bsmt Exposure",
    "BsmtFin Type 1",
    "BsmtFin Type 2",
    "Heating QC",
    "Kitchen Qual",
    "Fireplace Qu",
    "Garage Finish",
    "Garage Qual",
    "Garage Cond",
    "Pool QC",
    "Land Slope",
    "Functional",
    "Paved Drive",
}

STRUCTURAL_NA_THRESHOLD = 0.40


class SchemaAgent(BaseAgent):
    name = "schema_agent"
    version = "1.0.0"

    async def execute(self, input_data) -> SchemaOutput:
        df: pd.DataFrame = self._get_df(input_data)

        await self.emit(
            AgentStatus.PROGRESS, f"Validating {len(df.columns)}-column schema contract"
        )

        # Fuzzy match column names
        column_name_map = self._fuzzy_match_columns(list(df.columns))

        # Classify columns
        column_type_map: Dict[str, ColumnType] = {}
        columns_info: List[ColumnInfo] = []
        null_rates: Dict[str, float] = {}
        structural_na_candidates: List[str] = []
        num_continuous = num_discrete = num_ordinal = num_nominal = 0

        for col in df.columns:
            null_rate = df[col].isnull().sum() / len(df)
            null_rates[col] = round(null_rate, 4)
            unique_count = df[col].nunique()
            is_structural_na = null_rate > STRUCTURAL_NA_THRESHOLD

            if is_structural_na:
                structural_na_candidates.append(col)

            # Determine column type
            matched = column_name_map.get(col, col)
            if matched in NUMERIC_CONTINUOUS:
                ctype = ColumnType.NUMERIC_CONTINUOUS
                num_continuous += 1
            elif matched in NUMERIC_DISCRETE:
                ctype = ColumnType.NUMERIC_DISCRETE
                num_discrete += 1
            elif matched in ORDINAL_CATEGORICAL:
                ctype = ColumnType.ORDINAL_CATEGORICAL
                num_ordinal += 1
            else:
                ctype = ColumnType.NOMINAL_CATEGORICAL
                num_nominal += 1

            column_type_map[col] = ctype
            columns_info.append(
                ColumnInfo(
                    name=col,
                    matched_name=matched,
                    data_type=ctype,
                    null_rate=round(null_rate, 4),
                    unique_count=unique_count,
                    is_structural_na=is_structural_na,
                )
            )

        await self.emit(
            AgentStatus.PROGRESS,
            f"Column type classification complete: {num_continuous + num_discrete} numeric, "
            f"{num_ordinal + num_nominal} categorical",
        )

        await self.emit(
            AgentStatus.PROGRESS,
            f"Null rate analysis: {len(structural_na_candidates)} structural NA columns identified",
        )

        # Validate SalePrice
        sp = df["SalePrice"]
        sp_valid = sp.notna().all() and (sp > 0).all()
        sp_min, sp_max = sp.min(), sp.max()
        await self.emit(
            AgentStatus.PROGRESS,
            f"SalePrice validation: {sp.notna().sum():,} non-null, "
            f"range ${sp_min:,.0f}–${sp_max:,.0f} {'✓' if sp_valid else '✗'}",
        )

        # Compute confidence score
        total_cells = len(df) * len(df.columns)
        unexpected_nulls = sum(
            df[col].isnull().sum()
            for col in df.columns
            if col not in structural_na_candidates
        )
        confidence = 1.0 - (unexpected_nulls / total_cells) if total_cells > 0 else 0.0

        schema_report = SchemaReport(
            total_columns=len(df.columns),
            numeric_continuous=num_continuous,
            numeric_discrete=num_discrete,
            ordinal_categorical=num_ordinal,
            nominal_categorical=num_nominal,
            columns=columns_info,
            sale_price_valid=sp_valid,
        )

        await self.emit(
            AgentStatus.PROGRESS,
            f"Schema confidence: {confidence:.2f} | "
            f"{len(structural_na_candidates)} structural NAs | "
            f"{sum(1 for r in null_rates.values() if r > 0 and r <= STRUCTURAL_NA_THRESHOLD)} imputable columns",
        )

        return SchemaOutput(
            schema_report=schema_report,
            null_rates=null_rates,
            column_type_map=column_type_map,
            schema_confidence_score=round(confidence, 4),
            structural_na_candidates=structural_na_candidates,
            column_name_map=column_name_map,
            row_count=len(df),
        )

    def _fuzzy_match_columns(self, actual_cols: List[str]) -> Dict[str, str]:
        """Map actual column names to expected names using fuzzy matching."""
        mapping = {}
        for col in actual_cols:
            match, score, _ = process.extractOne(
                col, EXPECTED_COLUMNS, scorer=fuzz.ratio
            )
            if score > 80:
                mapping[col] = match
            else:
                mapping[col] = col
        return mapping

    def _get_df(self, input_data) -> pd.DataFrame:
        if isinstance(input_data, dict):
            agent = input_data.get("ingestion_agent")
            if hasattr(agent, "_df"):
                return agent._df
        if hasattr(input_data, "_df"):
            return input_data._df
        # Fallback: read from results
        raise ValueError("No DataFrame available from ingestion agent")
