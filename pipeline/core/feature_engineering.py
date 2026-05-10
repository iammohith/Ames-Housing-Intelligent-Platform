"""
Shared Feature Engineering Module
Single source of truth for all 12 engineered features.
Used by both feature_agent and predict endpoints to ensure consistency.
"""

from __future__ import annotations

import pandas as pd


FEATURE_DEFINITIONS = [
    (
        "TotalSF",
        "TotalBsmtSF + 1stFlrSF + 2ndFlrSF",
        "Combined livable space is primary value driver",
    ),
    (
        "PorchSF",
        "WoodDeckSF + OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch",
        "Outdoor usable space adds marginal value",
    ),
    ("HouseAge", "YrSold - YearBuilt", "Relative age prevents temporal leakage"),
    ("RemodAge", "YrSold - YearRemod/Add", "Recency of remodel affects value"),
    ("GarageAge", "YrSold - GarageYrBlt", "Garage condition proxy"),
    (
        "TotalBathrooms",
        "FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath",
        "Half-bath = 0.5 full-bath equivalent",
    ),
    ("HasPool", "PoolArea > 0", "Presence more predictive than raw area"),
    ("HasGarage", "GarageArea > 0", "Binary garage presence"),
    ("HasBasement", "TotalBsmtSF > 0", "Binary basement presence"),
    ("HasFireplace", "Fireplaces > 0", "Binary fireplace presence"),
    ("IsNew", "YearBuilt == YrSold", "New construction premium"),
    ("OverallScore", "OverallQual * OverallCond", "Quality × condition interaction"),
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer 12 domain features from raw Ames Housing data.
    
    Args:
        df: DataFrame with raw features (should have columns from Ames dataset)
        
    Returns:
        DataFrame with engineered features added
    """
    df = df.copy()
    
    # Area composites
    df["TotalSF"] = df["Total Bsmt SF"] + df["1st Flr SF"] + df["2nd Flr SF"]
    df["PorchSF"] = (
        df["Wood Deck SF"]
        + df["Open Porch SF"]
        + df["Enclosed Porch"]
        + df["3Ssn Porch"]
        + df["Screen Porch"]
    )
    
    # Age features (relative to sale year)
    df["HouseAge"] = df["Yr Sold"] - df["Year Built"]
    df["RemodAge"] = df["Yr Sold"] - df["Year Remod/Add"]
    df["GarageAge"] = df["Yr Sold"] - df["Garage Yr Blt"]
    df["GarageAge"] = df["GarageAge"].fillna(df["HouseAge"])
    
    # Bathroom composite
    df["TotalBathrooms"] = (
        df["Full Bath"]
        + 0.5 * df["Half Bath"]
        + df["Bsmt Full Bath"]
        + 0.5 * df["Bsmt Half Bath"]
    )
    
    # Binary presence flags
    df["HasPool"] = (df["Pool Area"] > 0).astype(int)
    df["HasGarage"] = (df["Garage Area"] > 0).astype(int)
    df["HasBasement"] = (df["Total Bsmt SF"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
    df["IsNew"] = (df["Year Built"] == df["Yr Sold"]).astype(int)
    
    # Interaction feature
    df["OverallScore"] = df["Overall Qual"] * df["Overall Cond"]
    
    return df


def engineer_features_from_dict(raw_inputs: dict, sale_year: int = 2010) -> dict:
    """
    Engineer features from a dictionary of raw input values.
    Used by predict endpoints to transform single-record inputs.
    
    Args:
        raw_inputs: Dict with keys matching Ames Housing feature names
        sale_year: Year of sale (for age calculations; defaults to 2010)
        
    Returns:
        Dict with engineered features added to raw_inputs
    """
    features = raw_inputs.copy()
    
    # Area composites
    features["TotalSF"] = (
        raw_inputs.get("Total Bsmt SF", 0)
        + raw_inputs.get("1st Flr SF", 0)
        + raw_inputs.get("2nd Flr SF", 0)
    )
    features["PorchSF"] = (
        raw_inputs.get("Wood Deck SF", 0)
        + raw_inputs.get("Open Porch SF", 0)
        + raw_inputs.get("Enclosed Porch", 0)
        + raw_inputs.get("3Ssn Porch", 0)
        + raw_inputs.get("Screen Porch", 0)
    )
    
    # Age features
    year_built = raw_inputs.get("Year Built", sale_year)
    features["HouseAge"] = max(sale_year - year_built, 0)
    
    features["RemodAge"] = max(
        sale_year - raw_inputs.get("Year Remod/Add", year_built), 0
    )
    
    garage_yr = raw_inputs.get("Garage Yr Blt", year_built)
    features["GarageAge"] = max(sale_year - garage_yr, 0)
    
    # Bathroom composite
    features["TotalBathrooms"] = (
        raw_inputs.get("Full Bath", 0)
        + 0.5 * raw_inputs.get("Half Bath", 0)
        + raw_inputs.get("Bsmt Full Bath", 0)
        + 0.5 * raw_inputs.get("Bsmt Half Bath", 0)
    )
    
    # Binary presence flags
    features["HasPool"] = 1 if raw_inputs.get("Pool Area", 0) > 0 else 0
    features["HasGarage"] = 1 if raw_inputs.get("Garage Area", 0) > 0 else 0
    features["HasBasement"] = 1 if raw_inputs.get("Total Bsmt SF", 0) > 0 else 0
    features["HasFireplace"] = 1 if raw_inputs.get("Fireplaces", 0) > 0 else 0
    features["IsNew"] = 1 if year_built == sale_year else 0
    
    # Interaction feature (CRITICAL: use Overall Cond, not hardcoded 5)
    overall_qual = raw_inputs.get("Overall Qual", 5)
    overall_cond = raw_inputs.get("Overall Cond", 5)
    features["OverallScore"] = overall_qual * overall_cond
    
    return features
