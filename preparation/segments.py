from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SENSOR_NAMES = ("co2", "pressure", "temperature", "ph")
SHARED_CHANNELS = {"pressure", "temperature", "ph"}


@dataclass
class PreparationConfig:
    # All selected channels must pass their applicable quality checks.
    channels: tuple = ("ph",)

    # Split segments when consecutive timestamps are too far apart.
    max_gap_seconds: float = 3.0

    # Initial analysis choices, not universal validity thresholds.
    min_duration_seconds: float = 120.0
    min_samples: int = 30

    split_on_mode_change: bool = True

    # Conservative initial policy.
    exclude_suspected_freezes: bool = True


def read_boolean_column(df, column):
    """
    Read exported audit flags safely.

    Avoid astype(bool) on strings:
    bool("False") would incorrectly evaluate to True.
    """
    if column not in df.columns:
        raise ValueError(f"Required audit flag is missing: {column}")

    text = df[column].astype("string").str.strip().str.lower()

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "1.0": True,
        "0.0": False,
    }

    result = text.map(mapping)

    if result.isna().any():
        examples = df.loc[result.isna(), column].head().tolist()
        raise ValueError(f"Unrecognized values in {column}: {examples}")

    return result.astype(bool)


def prepare_segments(input_path, output_dir, config):
    """
    Prepare one audited recording.

    Preserves:
      - original row order
      - original values
      - timestamps
      - audit flags

    Adds:
      - exclusion reasons
      - per-channel usable masks
      - masked value columns
      - retained segment IDs
    """
    if not config.channels:
        raise ValueError("Select at least one channel.")

    unknown = set(config.channels) - set(SENSOR_NAMES)
    if unknown:
        raise ValueError(f"Unknown channels: {sorted(unknown)}")

    if config.max_gap_seconds <= 0:
        raise ValueError("max_gap_seconds must be positive.")

    if config.min_duration_seconds < 0 or config.min_samples < 2:
        raise ValueError("Invalid minimum segment requirements.")

    df = pd.read_csv(input_path).reset_index(drop=True)

    if df.empty:
        raise ValueError("The annotated dataset is empty.")

    required = ["timestamp", "source_row"]
    required += [f"{name}_value" for name in config.channels]

    missing = [column for column in required if column not in df]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Keep the logger's original order.
    # Position is used to detect excluded rows between usable readings.
    df["preparation_position"] = np.arange(len(df))

    global_flags = [
        "flag_bad_timestamp",
        "flag_duplicate_timestamp",
        "flag_time_reversal",
        "flag_all_zero",
    ]

    # Build a table of exclusion reasons.
    reasons = pd.DataFrame(index=df.index)

    for flag in global_flags:
        reasons[flag] = read_boolean_column(df, flag)

    # Also check timestamps directly in case the CSV was edited.
    reasons["timestamp_parse_failed"] = df["timestamp"].isna()

    global_bad = reasons.any(axis=1)

    for name in config.channels:
        values = pd.to_numeric(df[f"{name}_value"], errors="coerce")

        invalid_flag = f"flag_{name}_invalid"
        reasons[invalid_flag] = read_boolean_column(df, invalid_flag)

        finite = pd.Series(
            np.isfinite(values.to_numpy(dtype=float)),
            index=df.index,
        )

        numeric_reason = f"{name}_nonfinite"
        reasons[numeric_reason] = ~finite

        channel_bad = global_bad | reasons[invalid_flag] | reasons[numeric_reason]

        if name in SHARED_CHANNELS and config.exclude_suspected_freezes:
            flag = "flag_shared_freeze_suspected"
            reasons[flag] = read_boolean_column(df, flag)
            channel_bad = channel_bad | reasons[flag]

        if name == "co2":
            flag = "flag_co2_at_limit"
            reasons[flag] = read_boolean_column(df, flag)
            channel_bad = channel_bad | reasons[flag]

        df[f"usable_{name}"] = ~channel_bad

        # New column only: original measurements are preserved.
        df[f"{name}_masked"] = values.where(~channel_bad)

    usable_columns = [f"usable_{name}" for name in config.channels]

    df["eligible_selected_channels"] = df[usable_columns].all(axis=1)

    df["exclusion_reasons"] = reasons.apply(
        lambda row: "; ".join(row.index[row].tolist()),
        axis=1,
    )

    # Mode is metadata, not proof of the physical reactor state.
    if "Mode" in df:
        modes = df["Mode"].fillna("Unknown").astype(str).str.strip().str.lower()
    else:
        modes = pd.Series("unknown", index=df.index)

    eligible = df["eligible_selected_channels"]
    dt = df["timestamp"].diff().dt.total_seconds()

    # Start a new segment after:
    # - an excluded row
    # - a time gap or non-increasing timestamp
    # - a logged mode change, if enabled
    boundary = (
        ~eligible.shift(fill_value=False)
        | dt.isna()
        | dt.le(0)
        | dt.gt(config.max_gap_seconds)
    )

    if config.split_on_mode_change:
        boundary = boundary | modes.ne(modes.shift())

    starts = eligible & boundary
    candidate_ids = starts.cumsum()

    df["segment_id"] = pd.Series(pd.NA, index=df.index, dtype="string")

    output_dir = Path(output_dir)
    segment_dir = output_dir / "segments"

    # Avoid mixing new outputs with an earlier preparation run.
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}\n"
            "Choose a new output directory for this run."
        )

    segment_dir.mkdir(parents=True, exist_ok=True)

    retained = []
    discarded = []

    for _, part in df.loc[eligible].groupby(candidate_ids.loc[eligible], sort=False):
        start = part["timestamp"].iloc[0]
        end = part["timestamp"].iloc[-1]
        duration = (end - start).total_seconds()
        intervals = part["timestamp"].diff().dt.total_seconds().dropna()

        details = {
            "start": start,
            "end": end,
            "duration_seconds": duration,
            "n_samples": len(part),
            "source_row_start": part["source_row"].iloc[0],
            "source_row_end": part["source_row"].iloc[-1],
            "channels": ", ".join(config.channels),
            "mode": (str(part["Mode"].iloc[0]) if "Mode" in part else "Unknown"),
            "gas_type": (
                str(part["gas_type"].iloc[0]) if "gas_type" in part else "Unknown"
            ),
            "median_dt_seconds": intervals.median(),
            "min_dt_seconds": intervals.min(),
            "max_dt_seconds": intervals.max(),
        }

        too_short = duration < config.min_duration_seconds
        too_few = len(part) < config.min_samples

        if too_short or too_few:
            failures = []

            if too_short:
                failures.append("below minimum duration")

            if too_few:
                failures.append("below minimum sample count")

            discarded.append(
                {
                    **details,
                    "reason": "; ".join(failures),
                }
            )
            continue

        segment_id = f"S{len(retained) + 1:03d}"
        df.loc[part.index, "segment_id"] = segment_id

        export = part.copy()
        export["segment_id"] = segment_id

        # Actual elapsed time, not row number.
        export["elapsed_seconds"] = (export["timestamp"] - start).dt.total_seconds()

        filename = f"{segment_id}.csv"
        export.to_csv(segment_dir / filename, index=False)

        retained.append(
            {
                "segment_id": segment_id,
                **details,
                "filename": filename,
            }
        )

    df["retained_for_segment_analysis"] = df["segment_id"].notna()

    df["preparation_status"] = "excluded_quality"
    df.loc[eligible, "preparation_status"] = "eligible_but_too_short"
    df.loc[
        df["retained_for_segment_analysis"],
        "preparation_status",
    ] = "retained"

    df.to_csv(
        output_dir / "prepared_timeline.csv",
        index=False,
    )

    manifest_columns = [
        "segment_id",
        "start",
        "end",
        "duration_seconds",
        "n_samples",
        "source_row_start",
        "source_row_end",
        "channels",
        "mode",
        "gas_type",
        "median_dt_seconds",
        "min_dt_seconds",
        "max_dt_seconds",
        "filename",
    ]

    pd.DataFrame(retained, columns=manifest_columns).to_csv(
        output_dir / "segment_manifest.csv",
        index=False,
    )

    pd.DataFrame(
        discarded,
        columns=[
            column
            for column in manifest_columns
            if column not in ("segment_id", "filename")
        ]
        + ["reason"],
    ).to_csv(
        output_dir / "discarded_short_segments.csv",
        index=False,
    )

    summary = (
        df["preparation_status"]
        .value_counts()
        .rename_axis("status")
        .reset_index(name="rows")
    )

    summary.to_csv(
        output_dir / "preparation_summary.csv",
        index=False,
    )

    print(f"\nSelected channels: {', '.join(config.channels)}")
    print(f"Retained segments: {len(retained)}")
    print(f"Short segments not retained: {len(discarded)}")
    print(summary.to_string(index=False))
    print(f"\nSaved to: {output_dir.resolve()}")

    return df, pd.DataFrame(retained)
