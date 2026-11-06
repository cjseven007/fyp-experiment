import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from characterization.analysis import (
    CharacterizationConfig,
    analyze_window,
)

from ph_benchmark.core import (
    SCENARIOS,
    alarms,
    block_slopes,
    bocpd_scores,
    event_metrics,
    generate,
    slope_scores,
    smooth,
)

WARMUP_SECONDS = 180
BLOCK_SECONDS = 30
SCENARIO_SECONDS = 1200


def characterize(data, window_seconds):
    """Recompute statistics for complete, non-overlapping windows."""
    rows = []

    for start in range(
        0,
        len(data) - window_seconds + 1,
        window_seconds,
    ):
        window = data.iloc[start : start + window_seconds]

        metrics, _ = analyze_window(
            window,
            CharacterizationConfig(),
        )

        rows.append(
            {
                "start_seconds": start,
                "window_seconds": window_seconds,
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def calibrate(segment_path, output_dir):
    """Validate the input and derive descriptive simulation inputs."""
    data = pd.read_csv(segment_path)

    required = {
        "timestamp",
        "ph_value",
        "usable_ph",
    }

    missing = required - set(data.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="raise",
    )

    data["ph_value"] = pd.to_numeric(
        data["ph_value"],
        errors="raise",
    )

    values = data["ph_value"].to_numpy(dtype=float)

    if len(values) < 120 or not np.isfinite(values).all():
        raise ValueError("Need at least 120 finite pH observations.")

    intervals = data["timestamp"].diff().dt.total_seconds().dropna()

    if not np.allclose(
        intervals,
        1.0,
        atol=0.05,
        rtol=0,
    ):
        raise ValueError(
            "This initial benchmark requires continuous " "1-second sampling."
        )

    usable = data["usable_ph"].astype(str).str.strip().str.lower().eq("true")

    if not usable.all():
        raise ValueError("Every input row must have usable_ph=True.")

    # Specific to the observed S005 recording precision.
    if not np.allclose(
        values * 100,
        np.round(values * 100),
        atol=1e-7,
        rtol=0,
    ):
        raise ValueError(
            "The assumed 0.01-pH recording grid does not fit "
            "this input. Review the calibration settings."
        )

    statistics = characterize(data, 60)

    statistics.to_csv(
        output_dir / "source_window_statistics.csv",
        index=False,
    )

    sensitivity = pd.concat(
        [characterize(data, seconds) for seconds in (30, 60, 120)],
        ignore_index=True,
    )

    sensitivity.to_csv(
        output_dir / "window_sensitivity.csv",
        index=False,
    )

    sensitivity_summary = sensitivity.groupby("window_seconds").agg(
        n_windows=("ph_mean", "size"),
        median_slope=(
            "slope_ph_per_minute",
            "median",
        ),
        median_residual_sd=("residual_sd", "median"),
        median_lag1=(
            "residual_lag1_correlation",
            "median",
        ),
        median_unchanged=(
            "unchanged_step_fraction",
            "median",
        ),
    )

    sensitivity_summary.to_csv(output_dir / "window_sensitivity_summary.csv")

    negative_slopes = statistics.loc[
        statistics["slope_ph_per_minute"] < -1e-8,
        "slope_ph_per_minute",
    ]

    positive_spreads = statistics.loc[
        statistics["residual_sd"] > 1e-10,
        "residual_sd",
    ]

    defined_correlations = statistics["residual_lag1_correlation"].dropna()

    if negative_slopes.empty:
        raise ValueError("No declining windows found. Review the scenario choice.")

    if positive_spreads.empty or defined_correlations.empty:
        raise ValueError(
            "Insufficient residual variation to derive the "
            "initial fluctuation settings."
        )

    profile = {
        "source_filename": Path(segment_path).name,
        "source_sha256": hashlib.sha256(Path(segment_path).read_bytes()).hexdigest(),
        "samples": len(data),
        "start": str(data["timestamp"].iloc[0]),
        "end": str(data["timestamp"].iloc[-1]),
        "start_ph": float(values[0]),
        "end_ph": float(values[-1]),
        "observed_grid_ph": 0.01,
        "decline_scale_ph_per_minute": float(-negative_slopes.median()),
        "positive_residual_sd_median": float(positive_spreads.median()),
        "residual_lag1_median": float(defined_correlations.median()),
        "unchanged_fraction_median": float(
            statistics["unchanged_step_fraction"].median()
        ),
        "note": (
            "Residual properties are descriptive. "
            "The latent AR fluctuation model is assumed, "
            "not identified. Recording precision is not "
            "certified sensor resolution."
        ),
    }

    if abs(profile["residual_lag1_median"]) >= 1:
        raise ValueError("AR correlation must lie strictly between -1 and 1.")

    (output_dir / "calibration.json").write_text(
        json.dumps(profile, indent=2),
        encoding="utf-8",
    )

    plot_source_context(data, output_dir)

    return data, profile


def plot_source_context(data, output_dir):
    """Plot all available channels without claiming they are validated."""
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(11, 9),
        sharex=True,
        constrained_layout=True,
    )

    labels = {
        "ph": "pH",
        "temperature": "Temperature (°C)",
        "pressure": "Pressure (barg)",
        "co2": "CO₂ (ppm)",
    }

    context = []
    minutes = np.arange(len(data)) / 60.0

    for ax, name in zip(
        axes,
        ("ph", "temperature", "pressure", "co2"),
    ):
        column = f"{name}_value"

        if column not in data:
            ax.text(
                0.5,
                0.5,
                f"{column} unavailable",
                transform=ax.transAxes,
                ha="center",
            )
            continue

        values = pd.to_numeric(
            data[column],
            errors="coerce",
        )

        ax.plot(minutes, values, linewidth=1)
        ax.set_ylabel(labels[name])
        ax.grid(alpha=0.2)

        context.append(
            {
                "channel": name,
                "start": values.iloc[0],
                "end": values.iloc[-1],
                "minimum": values.min(),
                "maximum": values.max(),
                "unique_values": values.nunique(),
                "nonfinite": int((~np.isfinite(values)).sum()),
            }
        )

    axes[0].set_title(
        "Source segment context\n"
        "pH-screened data; other channels require their own checks"
    )
    axes[-1].set_xlabel("Elapsed time (minutes)")

    fig.savefig(
        output_dir / "source_context.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

    pd.DataFrame(context).to_csv(
        output_dir / "channel_context.csv",
        index=False,
    )


def cases(profile, seeds):
    """Yield the same observations to every method in a comparison."""
    for scenario_number, scenario in enumerate(SCENARIOS):
        for multiplier in (0.0, 1.0, 2.0):
            for seed in seeds:
                generated = generate(
                    profile=profile,
                    scenario=scenario,
                    seed=seed + scenario_number * 100000,
                    noise_multiplier=multiplier,
                    n=SCENARIO_SECONDS,
                )

                yield scenario, multiplier, seed, generated


def detector_scores(filtered, scale, sigma):
    """Compute causal block features and both detector scores."""
    times, slopes = block_slopes(
        filtered,
        block=BLOCK_SECONDS,
    )

    normalized_slopes = slopes / scale

    scores = {
        "slope": slope_scores(normalized_slopes),
        "bocpd": bocpd_scores(
            normalized_slopes,
            sigma=sigma,
        ),
    }

    return times, scores


def select_smoothing(development_cases, output_dir):
    """Select one configuration per family using development RMSE."""
    grid = (
        [("raw", 1)]
        + [("ema", size) for size in (5, 15, 31, 61)]
        + [("wsg", size) for size in (15, 31, 61)]
    )

    rows = []

    for family, size in grid:
        for scenario, multiplier, seed, generated in development_cases:
            _, clean, observed, _ = generated

            filtered = smooth(observed, family, size)

            error = filtered[WARMUP_SECONDS:] - clean[WARMUP_SECONDS:]

            rows.append(
                {
                    "family": family,
                    "size": size,
                    "scenario": scenario,
                    "noise_multiplier": multiplier,
                    "seed": seed,
                    "rmse": float(np.sqrt(np.mean(error**2))),
                }
            )

    trials = pd.DataFrame(rows)

    trials.to_csv(
        output_dir / "smoothing_tuning_trials.csv",
        index=False,
    )

    summary = trials.groupby(
        ["family", "size"],
        as_index=False,
    )["rmse"].mean()

    summary.to_csv(
        output_dir / "smoothing_tuning_summary.csv",
        index=False,
    )

    chosen = {}

    for family, group in summary.groupby("family"):
        winner = group.loc[group["rmse"].idxmin()]
        chosen[family] = int(winner["size"])

    return chosen


def select_detectors(
    development_cases,
    chosen_smoothing,
    profile,
    output_dir,
):
    """
    Estimate working observation SD and select detector thresholds.

    Synthetic truth is used only during development/evaluation.
    It is never passed to a detector.
    """
    scale = profile["decline_scale_ph_per_minute"]

    settings = {}
    tuning_rows = []

    threshold_grids = {
        "slope": (0.25, 0.5, 0.75, 1.0, 1.5),
        "bocpd": (0.5, 0.7, 0.85, 0.95, 0.99),
    }

    for family, size in chosen_smoothing.items():
        slope_errors = []

        # Estimate a working feature-noise SD using development
        # scenarios without underlying change points.
        for scenario, _, _, generated in development_cases:
            if scenario not in ("stable", "decline"):
                continue

            _, clean, observed, _ = generated

            filtered = smooth(observed, family, size)

            times, estimated_slopes = block_slopes(filtered)
            _, true_slopes = block_slopes(clean)

            errors = (estimated_slopes - true_slopes) / scale

            slope_errors.extend(errors[times >= WARMUP_SECONDS])

        sigma = max(
            0.05,
            float(np.std(slope_errors, ddof=1)),
        )

        cached = []

        for scenario, multiplier, seed, generated in development_cases:
            _, _, observed, target = generated

            filtered = smooth(observed, family, size)

            times, scores = detector_scores(
                filtered,
                scale,
                sigma,
            )

            cached.append((target, times, scores))

        settings[family] = {
            "size": size,
            "normalized_observation_sd": sigma,
            "detectors": {},
        }

        for detector, thresholds in threshold_grids.items():
            options = []

            for threshold in thresholds:
                evaluations = []

                for target, times, scores in cached:
                    events = alarms(
                        times,
                        scores[detector],
                        threshold,
                    )

                    evaluations.append(event_metrics(events, target))

                results = pd.DataFrame(evaluations)

                miss_fraction = results["missed"].sum() / results["target_count"].sum()

                false_alarms_per_trace = results["false_alarms"].mean()

                mean_delay = results["delay_from_onset_seconds"].mean()

                delay_penalty = (
                    0.25 * mean_delay / 180.0 if np.isfinite(mean_delay) else 0.25
                )

                # Initial explicit tradeoff, not an operational
                # safety or false-alarm requirement.
                loss = 2.0 * miss_fraction + false_alarms_per_trace + delay_penalty

                option = {
                    "family": family,
                    "detector": detector,
                    "threshold": threshold,
                    "loss": loss,
                    "miss_rate": miss_fraction,
                    "false_alarms_per_trial": false_alarms_per_trace,
                    "mean_delay_seconds": mean_delay,
                }

                options.append(option)
                tuning_rows.append(option)

            winner = min(
                options,
                key=lambda item: item["loss"],
            )

            settings[family]["detectors"][detector] = winner["threshold"]

    pd.DataFrame(tuning_rows).to_csv(
        output_dir / "detector_tuning.csv",
        index=False,
    )

    (output_dir / "selected_settings.json").write_text(
        json.dumps(settings, indent=2),
        encoding="utf-8",
    )

    return settings


def evaluate_test_cases(
    profile,
    settings,
    test_seeds,
    output_dir,
):
    """Evaluate frozen settings on held-out synthetic realizations."""
    trace_dir = output_dir / "traces"
    trace_dir.mkdir()

    scale = profile["decline_scale_ph_per_minute"]

    smoothing_rows = []
    detection_rows = []
    diagnostic_rows = []

    for scenario, multiplier, seed, generated in cases(
        profile,
        test_seeds,
    ):
        times, clean, observed, target = generated

        trace = pd.DataFrame(
            {
                "elapsed_seconds": times,
                "clean_ph": clean,
                "observed_ph": observed,
                "target_onset_seconds": (np.nan if target is None else target[0]),
                "target_end_seconds": (np.nan if target is None else target[1]),
            }
        )

        # Inspect the first fixed realization of each condition.
        # This is not used to select model parameters.
        if seed == test_seeds[0]:
            diagnostic_data = pd.DataFrame(
                {
                    "timestamp": (
                        pd.Timestamp("2000-01-01") + pd.to_timedelta(times, unit="s")
                    ),
                    "ph_value": observed,
                }
            )

            diagnostic = characterize(diagnostic_data, 60)
            correlations = diagnostic["residual_lag1_correlation"].dropna()

            diagnostic_rows.append(
                {
                    "scenario": scenario,
                    "noise_multiplier": multiplier,
                    "median_residual_sd": diagnostic["residual_sd"].median(),
                    "median_lag1": (
                        correlations.median() if len(correlations) else np.nan
                    ),
                    "median_unchanged": diagnostic["unchanged_step_fraction"].median(),
                }
            )

        make_plot = seed == test_seeds[0] and multiplier == 1.0

        if make_plot:
            fig, axes = plt.subplots(
                2,
                1,
                figsize=(11, 6),
                sharex=True,
                constrained_layout=True,
            )

            axes[0].plot(
                times / 60,
                observed,
                color="gray",
                alpha=0.6,
                label="Observed synthetic",
            )

            axes[0].plot(
                times / 60,
                clean,
                color="black",
                label="Known clean target",
            )

        for family, setting in settings.items():
            filtered = smooth(
                observed,
                family,
                setting["size"],
            )

            trace[family] = filtered

            error = filtered - clean

            if target is None:
                transition = np.zeros(
                    len(times),
                    dtype=bool,
                )
            else:
                transition = (times >= target[0]) & (times <= target[1] + 180)

            transition &= times >= WARMUP_SECONDS

            smoothing_rows.append(
                {
                    "scenario": scenario,
                    "noise_multiplier": multiplier,
                    "seed": seed,
                    "family": family,
                    "rmse": float(np.sqrt(np.mean(error[WARMUP_SECONDS:] ** 2))),
                    "transition_rmse": (
                        float(np.sqrt(np.mean(error[transition] ** 2)))
                        if transition.any()
                        else np.nan
                    ),
                }
            )

            feature_times, scores = detector_scores(
                filtered,
                scale,
                setting["normalized_observation_sd"],
            )

            for detector, score in scores.items():
                events = alarms(
                    feature_times,
                    score,
                    setting["detectors"][detector],
                )

                metrics = event_metrics(events, target)

                detection_rows.append(
                    {
                        "scenario": scenario,
                        "noise_multiplier": multiplier,
                        "seed": seed,
                        "family": family,
                        "detector": detector,
                        "alarm_seconds": json.dumps(events),
                        **metrics,
                    }
                )

                trace.loc[
                    feature_times,
                    f"{family}_{detector}_score",
                ] = score

                trace[f"{family}_{detector}_alarm"] = np.isin(times, events)

                if make_plot and detector == "bocpd":
                    axes[1].plot(
                        feature_times / 60,
                        score,
                        label=family,
                    )

                    for event in events:
                        axes[0].axvline(
                            event / 60,
                            alpha=0.2,
                            linestyle=":",
                        )

            if make_plot and family != "raw":
                axes[0].plot(
                    times / 60,
                    filtered,
                    label=family,
                    linewidth=1,
                )

        trace.to_csv(
            trace_dir / f"{scenario}_noise{multiplier:g}_seed{seed}.csv",
            index=False,
        )

        if make_plot:
            if target is not None:
                for ax in axes:
                    ax.axvspan(
                        target[0] / 60,
                        max(target[1], target[0] + 1) / 60,
                        color="green",
                        alpha=0.2,
                    )

            axes[0].set_title(
                f"{scenario}: synthetic assumptions\n"
                "Dotted lines indicate BOCPD alarms"
            )
            axes[0].set_ylabel("pH")
            axes[1].set_ylabel("Recent-change probability")
            axes[1].set_xlabel("Elapsed time (minutes)")

            for ax in axes:
                ax.legend()
                ax.grid(alpha=0.2)

            fig.savefig(
                output_dir / f"{scenario}.png",
                dpi=160,
                bbox_inches="tight",
            )
            plt.close(fig)

    pd.DataFrame(diagnostic_rows).to_csv(
        output_dir / "synthetic_observed_diagnostics.csv",
        index=False,
    )

    save_test_summaries(
        smoothing_rows,
        detection_rows,
        output_dir,
    )


def save_test_summaries(
    smoothing_rows,
    detection_rows,
    output_dir,
):
    smoothing = pd.DataFrame(smoothing_rows)

    smoothing.to_csv(
        output_dir / "test_smoothing_trials.csv",
        index=False,
    )

    smoothing_summary = smoothing.groupby(
        ["scenario", "noise_multiplier", "family"]
    ).agg(
        n=("rmse", "size"),
        rmse_mean=("rmse", "mean"),
        rmse_sd=("rmse", "std"),
        transition_rmse_mean=(
            "transition_rmse",
            "mean",
        ),
    )

    smoothing_summary.to_csv(output_dir / "test_smoothing_summary.csv")

    detection = pd.DataFrame(detection_rows)

    detection.to_csv(
        output_dir / "test_detection_trials.csv",
        index=False,
    )

    detection_summary = (
        detection.groupby(
            [
                "scenario",
                "noise_multiplier",
                "family",
                "detector",
            ]
        )
        .agg(
            n=("seed", "size"),
            targets=("target_count", "sum"),
            detected=("detected", "sum"),
            missed=("missed", "sum"),
            false_alarms=("false_alarms", "sum"),
            negative_hours=(
                "negative_exposure_hours",
                "sum",
            ),
            median_delay_seconds=(
                "delay_from_onset_seconds",
                "median",
            ),
            mean_delay_seconds=(
                "delay_from_onset_seconds",
                "mean",
            ),
            median_delay_after_interval_seconds=(
                "delay_after_interval_seconds",
                "median",
            ),
        )
        .reset_index()
    )

    detection_summary["recall"] = detection_summary["detected"] / detection_summary[
        "targets"
    ].replace(0, np.nan)

    detection_summary["false_alarms_per_negative_hour"] = (
        detection_summary["false_alarms"] / detection_summary["negative_hours"]
    )

    detection_summary.to_csv(
        output_dir / "test_detection_summary.csv",
        index=False,
    )


def replay_real(data, profile, settings, output_dir):
    """
    Replay the real signal with frozen settings.

    No endpoint accuracy or clean-reference RMSE is calculated.
    """
    real = pd.DataFrame(
        {
            "timestamp": data["timestamp"],
            "elapsed_seconds": np.arange(len(data)),
            "ph_value": data["ph_value"],
        }
    )

    scale = profile["decline_scale_ph_per_minute"]
    event_rows = []

    fig, ax = plt.subplots(
        figsize=(11, 5),
        constrained_layout=True,
    )

    ax.plot(
        real["elapsed_seconds"] / 60,
        real["ph_value"],
        label="Recorded pH",
        color="gray",
    )

    for family, setting in settings.items():
        filtered = smooth(
            data["ph_value"].to_numpy(),
            family,
            setting["size"],
        )

        real[family] = filtered

        times, scores = detector_scores(
            filtered,
            scale,
            setting["normalized_observation_sd"],
        )

        for detector, score in scores.items():
            events = alarms(
                times,
                score,
                setting["detectors"][detector],
            )

            real.loc[
                times,
                f"{family}_{detector}_score",
            ] = score

            for event in events:
                event_rows.append(
                    {
                        "family": family,
                        "detector": detector,
                        "elapsed_seconds": event,
                        "timestamp": str(data["timestamp"].iloc[int(event)]),
                        "interpretation": ("Unvalidated statistical alarm"),
                    }
                )

        if family != "raw":
            ax.plot(
                real["elapsed_seconds"] / 60,
                filtered,
                label=family,
            )

    real.to_csv(
        output_dir / "real_replay.csv",
        index=False,
    )

    pd.DataFrame(
        event_rows,
        columns=[
            "family",
            "detector",
            "elapsed_seconds",
            "timestamp",
            "interpretation",
        ],
    ).to_csv(
        output_dir / "real_alarms.csv",
        index=False,
    )

    ax.set_xlabel("Elapsed time (minutes)")
    ax.set_ylabel("pH")
    ax.set_title("Real causal replay: no validated saturation label")
    ax.legend()
    ax.grid(alpha=0.2)

    fig.savefig(
        output_dir / "real_replay.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate pH-informed synthetic scenarios, compare "
            "causal smoothing and evaluate change detection."
        )
    )

    parser.add_argument(
        "--segment",
        required=True,
        help="Prepared S005.csv containing usable_ph.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="New or empty output directory.",
    )

    parser.add_argument(
        "--dev-reps",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--test-reps",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    if not (1 <= args.dev_reps <= 1000 and 1 <= args.test_reps <= 1000):
        raise ValueError("Repetition counts must be between 1 and 1000.")

    output_dir = Path(args.output)

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Choose a new or empty output directory.")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("[1/5] Checking and characterizing source segment...")

    data, profile = calibrate(
        args.segment,
        output_dir,
    )

    config = {
        "dev_seeds": list(range(1000, 1000 + args.dev_reps)),
        "test_seeds": list(range(10000, 10000 + args.test_reps)),
        "scenario_seconds": SCENARIO_SECONDS,
        "block_seconds": BLOCK_SECONDS,
        "metric_warmup_seconds": WARMUP_SECONDS,
        "matching_tolerance_seconds": 180,
        "noise_multipliers": [0, 1, 2],
        "hazard_per_block": 0.05,
        "prior_mean_normalized": -0.5,
        "prior_sd_normalized": 1.0,
        "assumptions": (
            "Synthetic endpoint-like transitions only. "
            "Real replay has no validated endpoint label "
            "and no detection accuracy metric."
        ),
    }

    (output_dir / "benchmark_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    development_cases = list(cases(profile, config["dev_seeds"]))

    print("[2/5] Selecting smoothing settings on development data...")

    chosen_smoothing = select_smoothing(
        development_cases,
        output_dir,
    )

    print("[3/5] Selecting detector settings on development data...")

    settings = select_detectors(
        development_cases,
        chosen_smoothing,
        profile,
        output_dir,
    )

    print("[4/5] Evaluating frozen settings on held-out simulations...")

    evaluate_test_cases(
        profile,
        settings,
        config["test_seeds"],
        output_dir,
    )

    print("[5/5] Replaying the real segment...")

    replay_real(
        data,
        profile,
        settings,
        output_dir,
    )

    print("\nSelected settings:")
    print(json.dumps(settings, indent=2))

    print(f"\nResults saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
