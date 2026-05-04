"""
Shared test fixtures for the Ames Housing Intelligence Platform.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

# Add pipeline to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))


@pytest.fixture
def sample_csv_path():
    return os.path.join(os.path.dirname(__file__), "..", "data", "AmesHousing.csv")


@pytest.fixture
def sample_df(sample_csv_path):
    """Load a small sample of the Ames dataset for testing."""
    if os.path.exists(sample_csv_path):
        df = pd.read_csv(sample_csv_path)
        return df
    # Fallback: generate synthetic data
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "Order": range(1, n + 1),
            "PID": [f"052630{i:04d}" for i in range(n)],
            "MS Zoning": np.random.choice(["RL", "RM", "RH"], n),
            "Lot Frontage": np.random.normal(70, 20, n),
            "Lot Area": np.random.normal(10000, 3000, n),
            "Neighborhood": np.random.choice(
                ["NAmes", "CollgCr", "OldTown", "NridgHt"], n
            ),
            "Overall Qual": np.random.randint(3, 9, n),
            "Overall Cond": np.random.randint(3, 8, n),
            "Year Built": np.random.randint(1950, 2010, n),
            "Year Remod/Add": np.random.randint(1960, 2010, n),
            "Gr Liv Area": np.random.normal(1500, 400, n),
            "Total Bsmt SF": np.random.normal(900, 300, n),
            "1st Flr SF": np.random.normal(1000, 300, n),
            "2nd Flr SF": np.random.choice([0, 400, 600, 800], n),
            "Full Bath": np.random.randint(1, 3, n),
            "Half Bath": np.random.randint(0, 2, n),
            "Bsmt Full Bath": np.random.randint(0, 2, n),
            "Bsmt Half Bath": np.random.randint(0, 1, n),
            "Garage Area": np.random.normal(400, 150, n),
            "Garage Yr Blt": np.random.randint(1950, 2010, n),
            "Fireplaces": np.random.randint(0, 3, n),
            "Pool Area": np.zeros(n),
            "Wood Deck SF": np.random.normal(50, 80, n).clip(0),
            "Open Porch SF": np.random.normal(30, 40, n).clip(0),
            "Enclosed Porch": np.zeros(n),
            "3Ssn Porch": np.zeros(n),
            "Screen Porch": np.zeros(n),
            "Yr Sold": np.random.choice([2006, 2007, 2008, 2009, 2010], n),
            "Mo Sold": np.random.randint(1, 13, n),
            "SalePrice": np.random.normal(180000, 50000, n).clip(50000),
            "Alley": [np.nan] * n,
            "Pool QC": [np.nan] * n,
            "Misc Feature": [np.nan] * n,
            "Fence": [np.nan] * n,
            "Fireplace Qu": np.random.choice(["Gd", "TA", np.nan], n),
            "Garage Type": np.random.choice(["Attchd", "Detchd", np.nan], n),
            "Garage Finish": np.random.choice(["Fin", "Unf", np.nan], n),
            "Garage Qual": np.random.choice(["TA", "Gd", np.nan], n),
            "Garage Cond": np.random.choice(["TA", np.nan], n),
            "Bsmt Qual": np.random.choice(["Gd", "TA", np.nan], n),
            "Bsmt Cond": np.random.choice(["TA", np.nan], n),
            "Bsmt Exposure": np.random.choice(["No", "Gd", np.nan], n),
            "BsmtFin Type 1": np.random.choice(["GLQ", "Unf", np.nan], n),
            "BsmtFin Type 2": np.random.choice(["Unf", np.nan], n),
            "Mas Vnr Type": np.random.choice(["BrkFace", "None", np.nan], n),
            "Mas Vnr Area": np.random.choice([0, 100, np.nan], n),
            "Electrical": np.random.choice(
                ["SBrkr", "FuseA", None], n, p=[0.9, 0.09, 0.01]
            ),
            "Exter Qual": np.random.choice(["TA", "Gd", "Ex"], n),
            "Exter Cond": np.random.choice(["TA", "Gd"], n),
            "Kitchen Qual": np.random.choice(["TA", "Gd", "Ex"], n),
            "Heating QC": np.random.choice(["TA", "Gd", "Ex"], n),
            "Sale Type": np.random.choice(["WD", "New", "COD"], n),
            "Sale Condition": np.random.choice(["Normal", "Abnorml"], n),
        }
    )


@pytest.fixture
def mock_event_bus():
    bus = MagicMock()
    bus.emit = AsyncMock()
    return bus


@pytest.fixture
def run_id():
    return "test-run-001"
