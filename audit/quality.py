import numpy as np
import pandas as pd

from .config import AuditConfig, SENSORS


def add_quality_flags(
    df: pd.DataFrame,
    config: AuditConfig,
) -> pd.DataFrame:
    out = df.copy()

    timestamp = out["timestamp"]
    dt = timestamp.diff().dt.total_seconds()
    out["dt_seconds"] = dt

    # Timestamp flags.
    out["flag_bad_timestamp"] = timestamp.isna()

    out["flag_duplicate_timestamp"] = timestamp.notna() & timestamp.duplicated(
        keep=False
    )

    out["flag_time_reversal"] = dt.lt(0)

    out["flag_time_gap"] = dt.gt(config.max_contiguous_gap_seconds)

    numeric_columns = [f"{name}_value" for name in SENSORS]

    out["flag_all_zero"] = out[numeric_columns].eq(0).all(axis=1)

    # Channel-specific validity checks.
    for name in SENSORS:
        values = out[f"{name}_value"]
        lower, upper = config.bounds[name]

        finite = pd.Series(
            np.isfinite(values.to_numpy(dtype=float)),
            index=out.index,
        )

        out[f"flag_{name}_invalid"] = ~finite | values.lt(lower) | values.gt(upper)

    # Exactly at the configured measurement ceiling.
    out["flag_co2_at_limit"] = out["co2_value"].eq(config.co2_upper_limit)

    # Look for simultaneous exact repeats across shared channels.
    shared_columns = [f"{name}_value" for name in config.shared_channels]

    shared_values = out[shared_columns]

    same_as_previous = shared_values.eq(shared_values.shift()).all(axis=1)

    # Zero-second intervals can occur due to timestamp resolution.
    # They contribute no elapsed duration.
    contiguous = (
        timestamp.notna()
        & timestamp.shift().notna()
        & dt.ge(0)
        & dt.le(config.max_contiguous_gap_seconds)
    )

    new_run = ~(same_as_previous & contiguous)
    run_id = new_run.cumsum()

    grouped_time = timestamp.groupby(run_id)
    run_start = grouped_time.transform("first")
    run_end = grouped_time.transform("last")

    duration = (run_end - run_start).dt.total_seconds()
    count = out.groupby(run_id)["source_row"].transform("size")

    out["shared_repeat_run_id"] = run_id
    out["shared_repeat_duration_seconds"] = duration

    out["flag_shared_freeze_suspected"] = duration.ge(
        config.freeze_min_seconds
    ) & count.ge(2)

    # Conservative candidate masks for later segment inspection.
    # These are NOT assertions that the remaining data are verified.
    timing_problem = (
        out["flag_bad_timestamp"]
        | out["flag_duplicate_timestamp"]
        | out["flag_time_reversal"]
    )

    for name in SENSORS:
        excluded = timing_problem | out["flag_all_zero"] | out[f"flag_{name}_invalid"]

        if name in config.shared_channels:
            excluded = excluded | out["flag_shared_freeze_suspected"]

        if name == "co2":
            excluded = excluded | out["flag_co2_at_limit"]

        out[f"candidate_{name}"] = ~excluded

    return out


def make_flag_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for column in df.columns:
        if column.startswith("flag_"):
            count = int(df[column].sum())

            rows.append(
                {
                    "flag": column,
                    "rows_flagged": count,
                    "percent_rows": (100.0 * count / len(df) if len(df) else 0.0),
                }
            )

    return pd.DataFrame(rows)
