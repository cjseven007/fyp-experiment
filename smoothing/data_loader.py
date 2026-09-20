from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

from .config import VARIABLES


REQUIRED_META_COLUMNS = ["Date", "Time"]


def _estimate_original_quantization_step(values: pd.Series) -> float | None:
    """Estimate resolution BEFORE timestamp interpolation creates new values."""
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(array) < 2:
        return None
    unique = np.unique(np.round(array, 8))
    if len(unique) < 2:
        return None
    diffs = np.diff(unique)
    diffs = diffs[diffs > 1e-9]
    if len(diffs) == 0:
        return None
    return float(np.percentile(diffs, 10))


def resolve_csv_files(path_or_directory: str | Path) -> List[Path]:
    """Return CSV files from either one CSV path or a directory.

    This makes the experiment automatically adapt from your current one CSV
    to 5, 10, or more cycles later. Simply add more CSV files to the folder.
    """

    path = Path(path_or_directory)

    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Expected a CSV file, got: {path}")
        return [path]

    if path.is_dir():
        files = sorted(path.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files found in directory: {path}")
        return files

    raise FileNotFoundError(f"Path does not exist: {path}")


def load_scada_csv(path: str | Path) -> pd.DataFrame:
    """Load and lightly preprocess one SCADA CSV.

    Important: this function does NOT smooth the signals. It only:
    1. validates expected columns,
    2. builds a timestamp,
    3. sorts rows,
    4. removes obvious all-zero startup rows,
    5. interpolates occasional irregular timestamp gaps onto a regular grid.
    """

    path = Path(path)
    df = pd.read_csv(path)

    signal_columns = [cfg.column for cfg in VARIABLES.values()]
    required = REQUIRED_META_COLUMNS + signal_columns
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        errors="coerce",
    )
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df.drop_duplicates(subset="timestamp", keep="first")

    # Convert signal columns to numbers. Invalid strings become NaN.
    for column in signal_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Estimate sensor resolution BEFORE any interpolation. Otherwise a missing
    # one-second row between 0.01 and 0.02 could create an artificial 0.015 and
    # incorrectly make the estimated resolution 0.005.
    original_quantization_steps = {
        column: _estimate_original_quantization_step(df[column])
        for column in signal_columns
    }

    # Remove startup region where ALL sensors are zero. In your supplied file,
    # this removes sensor-initialization zeros without deleting legitimate
    # later measurements merely because one variable is temporarily zero.
    all_zero = (df[signal_columns].fillna(0.0) == 0.0).all(axis=1)
    if (~all_zero).any():
        first_valid_position = int(np.flatnonzero((~all_zero).to_numpy())[0])
        df = df.iloc[first_valid_position:].copy()

    # Estimate normal sampling interval from the median timestamp difference.
    diffs = df["timestamp"].diff().dt.total_seconds().dropna()
    sampling_seconds = float(diffs.median()) if not diffs.empty else 1.0
    if sampling_seconds <= 0:
        sampling_seconds = 1.0

    # Reindex onto a regular time grid. This is useful because your current CSV
    # is nearly 1 Hz but contains a few 2-second gaps.
    freq = pd.to_timedelta(sampling_seconds, unit="s")
    regular_index = pd.date_range(df["timestamp"].iloc[0], df["timestamp"].iloc[-1], freq=freq)

    df = df.set_index("timestamp").reindex(regular_index)
    df.index.name = "timestamp"

    # Interpolate only numeric sensor values. Mode is forward/back-filled.
    df[signal_columns] = df[signal_columns].interpolate(method="time", limit_direction="both")

    if "Mode" in df.columns:
        df["Mode"] = df["Mode"].ffill().bfill()

    df = df.reset_index()
    df.attrs["sampling_seconds"] = sampling_seconds
    df.attrs["source_file"] = str(path)
    df.attrs["quantization_steps"] = original_quantization_steps
    return df


def load_all_datasets(path_or_directory: str | Path) -> List[pd.DataFrame]:
    """Load every CSV found at the provided file/folder path."""

    files = resolve_csv_files(path_or_directory)
    return [load_scada_csv(file) for file in files]
