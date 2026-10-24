from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class CharacterizationConfig:
    window_seconds: float = 60.0
    expected_dt_seconds: float = 1.0
    timing_tolerance_seconds: float = 0.05

    # Numerical tolerance only, not the instrument resolution.
    numerical_tolerance: float = 1e-10


def lag_correlation(values, lag=1, tolerance=1e-10):
    """Return correlation between observations separated by 'lag'."""
    values = np.asarray(values, dtype=float)

    if lag < 1 or len(values) <= lag + 2:
        return np.nan

    first = values[:-lag]
    second = values[lag:]

    if np.std(first) <= tolerance or np.std(second) <= tolerance:
        return np.nan

    return float(np.corrcoef(first, second)[0, 1])


def analyze_window(window, config):
    """
    Fit pH = intercept + slope * elapsed_seconds.

    Residuals are deviations from that fitted local line.
    They are not assumed to be pure measurement noise.
    """
    times = (
        (window["timestamp"] - window["timestamp"].iloc[0])
        .dt.total_seconds()
        .to_numpy()
    )

    values = window["ph_value"].to_numpy(dtype=float)

    # Center time to improve numerical conditioning.
    centered_time = times - times.mean()

    design = np.column_stack(
        [
            np.ones(len(times)),
            centered_time,
        ]
    )

    coefficients, _, _, _ = np.linalg.lstsq(
        design,
        values,
        rcond=None,
    )

    fitted = design @ coefficients
    residuals = values - fitted

    # Remove negligible floating-point residuals for exactly
    # constant or exactly linear data.
    if np.max(np.abs(residuals)) <= config.numerical_tolerance:
        residuals = np.zeros_like(residuals)

    residual_median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - residual_median)))

    residual_rms = float(np.sqrt(np.mean(residuals**2)))

    residual_sd = float(np.std(residuals, ddof=1))

    # Two fitted coefficients: intercept and slope.
    regression_residual_se = float(np.sqrt(np.sum(residuals**2) / (len(values) - 2)))

    unchanged = np.diff(values) == 0

    metrics = {
        "n_samples": len(values),
        "observed_span_seconds": float(times[-1] - times[0]),
        "ph_mean": float(np.mean(values)),
        "ph_min": float(np.min(values)),
        "ph_max": float(np.max(values)),
        "ph_range": float(np.ptp(values)),
        "n_unique_ph_values": int(len(np.unique(values))),
        "slope_ph_per_minute": float(coefficients[1] * 60.0),
        "residual_rms": residual_rms,
        "residual_sd": residual_sd,
        "regression_residual_se": regression_residual_se,
        "residual_mad": mad,
        # Normal-consistent scaling; descriptive when residuals
        # are rounded, dependent or non-Gaussian.
        "residual_mad_scaled": 1.4826 * mad,
        "residual_lag1_correlation": lag_correlation(
            residuals,
            lag=1,
            tolerance=config.numerical_tolerance,
        ),
        "unchanged_step_fraction": float(np.mean(unchanged)),
        "constant_recorded_window": bool(np.ptp(values) == 0),
    }

    detail = pd.DataFrame(
        {
            "timestamp": window["timestamp"].to_numpy(),
            "elapsed_seconds": times,
            "ph_value": values,
            "local_linear_trend": fitted,
            "residual": residuals,
        }
    )

    if "source_row" in window.columns:
        detail.insert(
            0,
            "source_row",
            window["source_row"].to_numpy(),
        )

    return metrics, detail


def characterize_segments(prepared_folder, config):
    """
    Read segment_manifest.csv and its segment files.

    Returns:
        statistics: accepted-window metrics
        skipped: rejected-window log
        details: accepted-window data for plotting
    """
    folder = Path(prepared_folder)

    if config.window_seconds <= 0:
        raise ValueError("window_seconds must be positive.")

    if config.expected_dt_seconds <= 0:
        raise ValueError("expected_dt_seconds must be positive.")

    if config.timing_tolerance_seconds < 0:
        raise ValueError("timing tolerance cannot be negative.")

    samples_float = config.window_seconds / config.expected_dt_seconds
    expected_samples = int(round(samples_float))

    if not np.isclose(samples_float, expected_samples):
        raise ValueError(
            "Window duration must be a whole multiple "
            "of the expected sampling interval."
        )

    if expected_samples < 4:
        raise ValueError("Each window must contain at least 4 samples.")

    manifest = pd.read_csv(
        folder / "segment_manifest.csv",
        dtype={"segment_id": "string"},
    )

    required_manifest = {"segment_id", "filename"}
    if not required_manifest.issubset(manifest.columns):
        raise ValueError("The segment manifest lacks required columns.")

    if manifest["segment_id"].duplicated().any():
        raise ValueError("Duplicate segment IDs found in the manifest.")

    statistics = []
    skipped = []
    details = {}

    for _, record in manifest.iterrows():
        segment_id = str(record["segment_id"])

        segment_path = folder / "segments" / str(record["filename"])

        segment = pd.read_csv(segment_path)

        required = {"timestamp", "ph_value"}
        if not required.issubset(segment.columns):
            raise ValueError(
                f"{segment_path.name} must contain timestamp and ph_value."
            )

        segment["timestamp"] = pd.to_datetime(segment["timestamp"], errors="coerce")
        segment["ph_value"] = pd.to_numeric(segment["ph_value"], errors="coerce")

        def log_skip(window_id, reason, n_samples, start=pd.NaT):
            skipped.append(
                {
                    "segment_id": segment_id,
                    "window_id": window_id,
                    "window_start": start,
                    "n_samples": n_samples,
                    "reason": reason,
                }
            )

        if segment.empty:
            log_skip("", "empty_segment", 0)
            continue

        if segment["timestamp"].isna().any():
            log_skip(
                "",
                "segment_contains_invalid_timestamps",
                len(segment),
            )
            continue

        segment_dt = segment["timestamp"].diff().dt.total_seconds().dropna()

        if segment_dt.le(0).any():
            log_skip(
                "",
                "segment_contains_duplicate_or_reversed_timestamps",
                len(segment),
            )
            continue

        # Preserve actual time. Never treat row number as elapsed time.
        origin = segment["timestamp"].iloc[0]

        elapsed = (segment["timestamp"] - origin).dt.total_seconds()

        # Windows: [0,60), [60,120), etc.
        window_numbers = np.floor(elapsed / config.window_seconds).astype(int)

        last_window = int(window_numbers.max())

        for window_number in range(last_window + 1):
            window_id = f"{segment_id}_W{window_number + 1:03d}"

            start = origin + pd.Timedelta(seconds=window_number * config.window_seconds)

            window = segment.loc[window_numbers.eq(window_number)].copy()

            if len(window) != expected_samples:
                log_skip(
                    window_id,
                    "incomplete_window_or_missing_samples",
                    len(window),
                    start,
                )
                continue

            # Confirm every timestamp lies on the expected local grid.
            offsets = (window["timestamp"] - start).dt.total_seconds().to_numpy()

            expected_offsets = np.arange(expected_samples) * config.expected_dt_seconds

            regular = np.allclose(
                offsets,
                expected_offsets,
                atol=config.timing_tolerance_seconds,
                rtol=0.0,
            )

            if not regular:
                log_skip(
                    window_id,
                    "irregular_sampling",
                    len(window),
                    start,
                )
                continue

            values = window["ph_value"].to_numpy(dtype=float)

            if not np.isfinite(values).all():
                log_skip(
                    window_id,
                    "nonfinite_ph_values",
                    len(window),
                    start,
                )
                continue

            # Preparation normally splits by mode. Check again.
            if "Mode" in window:
                modes = (
                    window["Mode"].fillna("Unknown").astype(str).str.strip().str.lower()
                )

                if modes.nunique() > 1:
                    log_skip(
                        window_id,
                        "mode_change_inside_window",
                        len(window),
                        start,
                    )
                    continue

            metrics, detail = analyze_window(window, config)

            statistics.append(
                {
                    "segment_id": segment_id,
                    "window_id": window_id,
                    "window_start": start,
                    "window_end_exclusive": start
                    + pd.Timedelta(seconds=config.window_seconds),
                    "last_observed_timestamp": window["timestamp"].iloc[-1],
                    "mode": record.get("mode", "Unknown"),
                    "gas_type": record.get("gas_type", "Unknown"),
                    **metrics,
                }
            )

            details[window_id] = detail

    skipped_columns = [
        "segment_id",
        "window_id",
        "window_start",
        "n_samples",
        "reason",
    ]

    return (
        pd.DataFrame(statistics),
        pd.DataFrame(skipped, columns=skipped_columns),
        details,
    )


def summarize_by_segment(statistics):
    """Describe variation between windows within each segment."""
    if statistics.empty:
        return pd.DataFrame()

    return (
        statistics.groupby(
            ["segment_id", "mode", "gas_type"],
            dropna=False,
            sort=False,
        )
        .agg(
            n_windows=("window_id", "size"),
            median_slope_ph_per_minute=("slope_ph_per_minute", "median"),
            median_residual_sd=("residual_sd", "median"),
            q25_residual_sd=(
                "residual_sd",
                lambda values: values.quantile(0.25),
            ),
            q75_residual_sd=(
                "residual_sd",
                lambda values: values.quantile(0.75),
            ),
            median_mad_scaled=("residual_mad_scaled", "median"),
            median_lag1_correlation=("residual_lag1_correlation", "median"),
            median_unchanged_fraction=("unchanged_step_fraction", "median"),
            constant_windows=("constant_recorded_window", "sum"),
        )
        .reset_index()
    )
