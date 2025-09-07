"""Data utilities for Jupyter notebooks."""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


def load_sample_data(n_rows: int = 100) -> pd.DataFrame:
    """Generate sample data for demonstrations.

    Args:
        n_rows: Number of rows to generate

    Returns:
        DataFrame with sample data
    """
    np.random.seed(42)

    data = {
        "id": range(1, n_rows + 1),
        "value": np.random.randn(n_rows) * 100 + 500,
        "category": np.random.choice(["A", "B", "C", "D"], n_rows),
        "timestamp": pd.date_range("2024-01-01", periods=n_rows, freq="H"),
        "is_valid": np.random.choice([True, False], n_rows, p=[0.9, 0.1]),
    }

    return pd.DataFrame(data)


def load_external_data(
    filepath: Union[str, Path], file_type: Optional[str] = None
) -> pd.DataFrame:
    """Load data from external file.

    Args:
        filepath: Path to data file
        file_type: Type of file ('csv', 'json', 'excel', 'parquet')
                  If None, inferred from extension

    Returns:
        Loaded DataFrame
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if file_type is None:
        file_type = filepath.suffix[1:].lower()

    loaders = {
        "csv": pd.read_csv,
        "json": pd.read_json,
        "excel": pd.read_excel,
        "xlsx": pd.read_excel,
        "parquet": pd.read_parquet,
    }

    if file_type not in loaders:
        raise ValueError(f"Unsupported file type: {file_type}")

    return loaders[file_type](filepath)


def summarize_data(df: pd.DataFrame) -> dict:
    """Generate summary statistics for DataFrame.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "memory_usage": df.memory_usage(deep=True).sum() / 1024**2,  # MB
    }

    # Add numeric column statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary["numeric_stats"] = df[numeric_cols].describe().to_dict()

    return summary


def clean_data(
    df: pd.DataFrame, drop_nulls: bool = False, drop_duplicates: bool = True
) -> pd.DataFrame:
    """Basic data cleaning operations.

    Args:
        df: Input DataFrame
        drop_nulls: Whether to drop rows with null values
        drop_duplicates: Whether to drop duplicate rows

    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()

    if drop_duplicates:
        df_clean = df_clean.drop_duplicates()

    if drop_nulls:
        df_clean = df_clean.dropna()

    return df_clean
