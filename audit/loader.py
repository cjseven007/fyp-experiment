from pathlib import Path

import pandas as pd

from .config import SENSORS


def load_recording(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)

    required = ["Date", "Time", *SENSORS.values()]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # One-based data row number, excluding the CSV header.
    df.insert(0, "source_row", range(1, len(df) + 1))

    df["timestamp"] = pd.to_datetime(
        df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip(),
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )

    # Keep original sensor columns untouched.
    # Add numeric copies for analysis.
    for name, column in SENSORS.items():
        df[f"{name}_value"] = pd.to_numeric(df[column], errors="coerce")

    df["source_file"] = path.name

    return df
