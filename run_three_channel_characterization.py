"""Characterise pH, temperature and pressure on the same time windows."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Name in the prepared CSV: (display label, unit, colour)
CHANNELS = {
    "ph": ("pH", "pH", "tab:blue"),
    "temperature": ("Temperature", "°C", "tab:orange"),
    "pressure": ("Pressure", "barg", "tab:green"),
}

NUMERICAL_TOLERANCE = 1e-10


def read_segment(path):
    """Read one multivariate prepared segment without sorting or deleting rows."""
    df = pd.read_csv(path)

    required = {"timestamp", "segment_id"}
    for channel in CHANNELS:
        required.update({f"{channel}_value", f"usable_{channel}"})

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"Missing columns: {missing}\n"
            "Use a segment from the pH + temperature + pressure preparation."
        )

    if df.empty:
        raise ValueError("The segment is empty.")

    if df["segment_id"].isna().any() or df["segment_id"].nunique() != 1:
        raise ValueError("The input must contain exactly one retained segment.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("The segment contains an invalid timestamp.")

    dt = df["timestamp"].diff().dt.total_seconds().dropna()
    if (dt <= 0).any():
        raise ValueError(
            "Timestamps are duplicated or reversed. Check the prepared segment."
        )

    for channel in CHANNELS:
        value_col = f"{channel}_value"
        usable_col = f"usable_{channel}"

        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

        # Read text flags safely: bool("False") would incorrectly return True.
        flag = (
            df[usable_col]
            .astype("string")
            .str.strip()
            .str.lower()
            .map(
                {
                    "true": True,
                    "false": False,
                    "1": True,
                    "0": False,
                    "1.0": True,
                    "0.0": False,
                }
            )
        )
        if flag.isna().any():
            raise ValueError(f"Missing or unrecognised flag in {usable_col}.")

        df[usable_col] = flag.astype(bool)

    df["elapsed_seconds"] = (
        df["timestamp"] - df["timestamp"].iloc[0]
    ).dt.total_seconds()

    return df


def measure_channel(values, seconds):
    """Fit a local line and describe the recorded values and their residuals."""
    y = np.asarray(values, dtype=float)
    t = np.asarray(seconds, dtype=float)

    # Least-squares slope with an intercept, using actual observation times.
    x = t - t.mean()
    slope_per_second = np.dot(x, y - y.mean()) / np.dot(x, x)
    fitted = y.mean() + slope_per_second * x
    residual = y - fitted

    # Avoid reporting floating-point arithmetic as real variation.
    if np.max(np.abs(residual)) <= NUMERICAL_TOLERANCE:
        residual = np.zeros_like(residual)

    previous, current = residual[:-1], residual[1:]
    if previous.std() > NUMERICAL_TOLERANCE and current.std() > NUMERICAL_TOLERANCE:
        lag1 = float(np.corrcoef(previous, current)[0, 1])
    else:
        lag1 = np.nan  # Undefined, not zero.

    mad = np.median(np.abs(residual - np.median(residual)))

    metrics = {
        "mean_value": float(y.mean()),
        "min_value": float(y.min()),
        "max_value": float(y.max()),
        "range_value": float(np.ptp(y)),
        "n_unique_values": int(np.unique(y).size),
        "slope_per_minute": float(60 * slope_per_second),
        "residual_sd": float(residual.std(ddof=1)),
        "residual_mad_scaled": float(1.4826 * mad),
        "residual_lag1": lag1,
        "unchanged_step_fraction": float(np.mean(np.diff(y) == 0)),
        "constant_window": bool(np.ptp(y) == 0),
    }
    return metrics, fitted, residual


def analyse_segment(df, window_seconds, expected_dt, timing_tolerance):
    """Accept a window only if timing and quality pass for all three channels."""
    if window_seconds <= 0 or expected_dt <= 0 or timing_tolerance < 0:
        raise ValueError("Check window duration, sampling interval and tolerance.")

    ratio = window_seconds / expected_dt
    expected_n = round(ratio)
    if not np.isclose(ratio, expected_n) or expected_n < 4:
        raise ValueError("Each window must contain a whole number of ≥4 samples.")

    origin = df["timestamp"].iloc[0]
    segment_id = str(df["segment_id"].iloc[0])
    window_number = np.floor(df["elapsed_seconds"] / window_seconds).astype(int)
    groups = dict(tuple(df.groupby(window_number, sort=False)))

    stats_rows, detail_frames, skipped_rows = [], [], []

    for number in range(int(window_number.max()) + 1):
        window = groups.get(number, df.iloc[0:0])
        start = origin + pd.Timedelta(seconds=number * window_seconds)

        info = {
            "segment_id": segment_id,
            "window_id": f"{segment_id}_W{number + 1:03d}",
            "window_number": number + 1,
            "window_start": start,
            "window_end_exclusive": start + pd.Timedelta(seconds=window_seconds),
            "n_samples": len(window),
        }

        offsets = (window["timestamp"] - start).dt.total_seconds().to_numpy()
        expected_offsets = np.arange(expected_n) * expected_dt

        if len(window) != expected_n:
            reason = "incomplete_window_or_missing_samples"
        elif not np.allclose(offsets, expected_offsets, atol=timing_tolerance, rtol=0):
            reason = "irregular_sampling"
        elif "Mode" in window and (
            window["Mode"]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .str.lower()
            .nunique()
            > 1
        ):
            reason = "mode_change_inside_window"
        else:
            bad_channels = [
                ch
                for ch in CHANNELS
                if not window[f"usable_{ch}"].all()
                or not np.isfinite(window[f"{ch}_value"].to_numpy()).all()
            ]
            reason = (
                "unusable_channels: " + ", ".join(bad_channels) if bad_channels else ""
            )

        if reason:
            skipped_rows.append({**info, "reason": reason})
            continue

        for channel, (label, unit, _) in CHANNELS.items():
            values = window[f"{channel}_value"].to_numpy(dtype=float)
            metrics, fitted, residual = measure_channel(values, offsets)

            stats_rows.append(
                {
                    **info,
                    "channel": channel,
                    "unit": unit,
                    **metrics,
                }
            )

            detail = pd.DataFrame(
                {
                    "timestamp": window["timestamp"].to_numpy(),
                    "elapsed_seconds": window["elapsed_seconds"].to_numpy(),
                    "window_id": info["window_id"],
                    "channel": channel,
                    "observed": values,
                    "local_linear_trend": fitted,
                    "residual": residual,
                }
            )
            if "source_row" in window:
                detail["source_row"] = window["source_row"].to_numpy()

            detail_frames.append(detail)

    stats = pd.DataFrame(stats_rows)
    details = (
        pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    )
    skipped = pd.DataFrame(
        skipped_rows,
        columns=[
            "segment_id",
            "window_id",
            "window_number",
            "window_start",
            "window_end_exclusive",
            "n_samples",
            "reason",
        ],
    )
    return stats, details, skipped


def make_plots(df, stats, details, skipped, window_seconds, output_dir):
    """Show the full recording, accepted fits, and aligned window slopes."""
    segment_id = str(df["segment_id"].iloc[0])
    minutes = df["elapsed_seconds"] / 60

    fig, axes = plt.subplots(
        3, 1, figsize=(11, 8), sharex=True, constrained_layout=True
    )

    for ax, (channel, (label, unit, colour)) in zip(axes, CHANNELS.items()):
        ax.plot(
            minutes,
            df[f"{channel}_value"],
            color=colour,
            linewidth=1.1,
            label="Recorded",
        )

        if not details.empty:
            channel_data = details.loc[details["channel"].eq(channel)]
            for i, (_, part) in enumerate(
                channel_data.groupby("window_id", sort=False)
            ):
                ax.plot(
                    part["elapsed_seconds"] / 60,
                    part["local_linear_trend"],
                    color="black",
                    linewidth=1.4,
                    linestyle="--",
                    label="Local fit" if i == 0 else None,
                )

        for i, row in enumerate(skipped.itertuples()):
            left = (row.window_number - 1) * window_seconds / 60
            right = min(left + window_seconds / 60, minutes.iloc[-1])
            ax.axvspan(
                left,
                right,
                color="grey",
                alpha=0.18,
                label="Excluded from statistics" if i == 0 else None,
            )

        ax.set_ylabel(f"{label} ({unit})" if channel != "ph" else "pH")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Minutes from segment start")
    fig.suptitle(f"{segment_id}: recorded signals and local linear fits")
    fig.savefig(output_dir / "01_signals_and_fits.png", dpi=200)
    plt.close(fig)

    if stats.empty:
        return

    fig, axes = plt.subplots(
        3, 1, figsize=(11, 7), sharex=True, constrained_layout=True
    )

    for ax, (channel, (label, unit, colour)) in zip(axes, CHANNELS.items()):
        part = stats.loc[stats["channel"].eq(channel)]
        ax.scatter(part["window_number"], part["slope_per_minute"], color=colour, s=35)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel(f"{label} slope\n({unit}/min)")
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("Window number from segment start")
    fig.suptitle(
        f"{segment_id}: slopes from the same {window_seconds:g}-second windows"
    )
    fig.savefig(output_dir / "02_aligned_slopes.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--window-seconds", type=float, default=60)
    parser.add_argument("--expected-dt", type=float, default=1)
    parser.add_argument("--timing-tolerance", type=float, default=0.05)
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Choose a new or empty output folder: {args.output}")

    df = read_segment(args.segment)
    stats, details, skipped = analyse_segment(
        df, args.window_seconds, args.expected_dt, args.timing_tolerance
    )

    args.output.mkdir(parents=True, exist_ok=True)
    skipped.to_csv(args.output / "skipped_windows.csv", index=False)
    make_plots(df, stats, details, skipped, args.window_seconds, args.output)

    if stats.empty:
        print("No jointly usable windows. See skipped_windows.csv and the signal plot.")
        return

    stats.to_csv(args.output / "window_statistics.csv", index=False)
    details.to_csv(args.output / "window_details.csv", index=False)

    summary = (
        stats.groupby(["segment_id", "channel", "unit"], sort=False)
        .agg(
            n_windows=("window_id", "size"),
            median_slope_per_minute=("slope_per_minute", "median"),
            median_residual_sd=("residual_sd", "median"),
            q25_residual_sd=("residual_sd", lambda x: x.quantile(0.25)),
            q75_residual_sd=("residual_sd", lambda x: x.quantile(0.75)),
            median_lag1=("residual_lag1", "median"),
            n_defined_lag1=("residual_lag1", "count"),
            median_unchanged_fraction=("unchanged_step_fraction", "median"),
            constant_windows=("constant_window", "sum"),
        )
        .reset_index()
    )
    summary.to_csv(args.output / "channel_summary.csv", index=False)

    print(f"Accepted aligned windows: {stats['window_id'].nunique()}")
    print(f"Skipped windows: {len(skipped)}")
    print(summary.round(5).to_string(index=False))
    print(f"Saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
