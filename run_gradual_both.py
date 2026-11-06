import argparse
import ast
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import lfilter

from run_multichannel_pilot import (
    CH, GRID, BLOCK, TOL,
    read_segment, characterize, calibrate,
    filtered, features, scores, decide,
)
from run_confirmation import confirm


PHASES = ("absorption", "desorption")
METHODS = ("slope", "uni", "multi")
CASES = ("continued", "abrupt", "gradual", "pause")


def paired_cases(profile, seed, phase):
    """Four traces with identical operating context and measurement noise."""
    rng = np.random.default_rng([seed, int(phase == "desorption")])

    t = np.arange(720)                   # 12 minutes, 1 Hz
    onset = int(rng.integers(250, 351))  # start of the introduced change
    width = int(rng.integers(60, 121))   # gradual transition or pause duration

    direction = -1 if phase == "absorption" else 1

    # S005 informs the magnitude. The upward desorption-like rate is
    # an explicit simulation assumption, not measured desorption kinetics.
    slope = (
        direction * profile["decline"]
        * rng.uniform(0.75, 1.25) / 60
    )
    ph0 = profile["start"][0] + rng.uniform(-0.03, 0.03)

    # Temperature and pressure are context channels in this experiment.
    # Neither is deliberately made to announce the pH endpoint.
    temperature = (
        profile["start"][1]
        + profile["heat"] * rng.uniform(0.6, 1.15) * t / 60
        + 0.05 * profile["span"][1]
        * np.sin(2 * np.pi * t / rng.uniform(100, 220))
    )

    pressure = (
        profile["pressure"]
        + 0.12 * profile["span"][2]
        * np.sin(
            2 * np.pi * t / rng.uniform(70, 170)
            + rng.uniform(0, 2 * np.pi)
        )
    )

    # Draw fluctuations ONCE and reuse them across all four cases.
    multiplier = float(rng.choice([0.0, 0.5, 1.0]))
    noise = np.empty((len(t), 3))

    for j, (sd, rho) in enumerate(zip(profile["sd"], profile["rho"])):
        sigma = multiplier * sd
        innovations = (
            rng.normal(size=len(t))
            * sigma * np.sqrt(1 - rho * rho)
        )
        innovations[0] = rng.normal() * sigma
        noise[:, j] = lfilter([1.0], [1.0, -rho], innovations)

    # clean_pH = starting_pH + initial_slope * effective_time.
    #
    # For gradual flattening, the slope decreases linearly in magnitude
    # from its initial value to zero over 'width' seconds.
    u = np.clip(t - onset, 0, width)

    effective_time = {
        "continued": t,
        "abrupt": np.minimum(t, onset),
        "gradual": np.minimum(t, onset) + u - u**2 / (2 * width),
        "pause": np.minimum(t, onset) + np.maximum(t - onset - width, 0),
    }

    for case in CASES:
        clean = np.column_stack([
            ph0 + slope * effective_time[case],
            temperature,
            pressure,
        ])
        obs = np.round((clean + noise) / GRID) * GRID

        if case == "abrupt":
            transition_start = onset
            plateau_start = onset
        elif case == "gradual":
            transition_start = onset
            plateau_start = onset + width
        else:
            # A temporary flat section is not a target endpoint.
            transition_start = None
            plateau_start = None

        yield (
            case, clean, obs, transition_start, plateau_start,
            onset, width, multiplier,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Paired abrupt versus gradual two-phase pH pilot."
    )
    parser.add_argument("--segment", required=True, type=Path)
    parser.add_argument("--pilot", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=800000)
    args = parser.parse_args()

    if args.reps < 2 or args.seed_start < 0:
        parser.error("Use at least two repetitions and nonnegative seeds.")

    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Choose a new or empty output directory.")

    if BLOCK != 30 or TOL != 180:
        parser.error("This protocol assumes 30-second blocks and a 180-second tolerance.")

    seeds = range(args.seed_start, args.seed_start + args.reps)

    # Check that the user supplied the TWO-PHASE results folder.
    old = pd.read_csv(
        args.pilot / "test_events.csv",
        usecols=["phase", "seed"],
    )
    if set(old["phase"]) != set(PHASES):
        parser.error("Use multichannel_both_v1, not the absorption-only folder.")

    if set(seeds).intersection(old["seed"]):
        parser.error("Use seeds outside the original pilot test range.")

    # Reconstruct the S005 generator profile and verify source consistency.
    _, real = read_segment(args.segment)
    real_stats = characterize(real)
    original_stats = pd.read_csv(args.pilot / "real_windows.csv")

    metric_cols = ["slope_per_min", "residual_sd", "lag1", "unchanged"]

    def ordered(frame):
        return (
            frame.sort_values(["channel", "window"])[metric_cols]
            .to_numpy()
        )

    current = ordered(real_stats)
    original = ordered(original_stats)

    if current.shape != original.shape or not np.allclose(
        current, original, equal_nan=True
    ):
        parser.error("The source segment does not match the selected pilot.")

    profile = calibrate(real, real_stats)

    # Freeze WSG to isolate the effect of transition shape.
    # Compare all three detectors using their own previously tuned thresholds.
    cfg = pd.read_csv(args.pilot / "frozen_settings.csv")
    cfg = (
        cfg.loc[cfg["family"].eq("wsg")]
        .set_index("method")
        .loc[list(METHODS)]
    )

    for col in ["size", "center", "scale"]:
        if cfg[col].nunique() != 1:
            parser.error("Expected shared WSG size and scaling across methods.")

    first = cfg.iloc[0]
    size = int(first["size"])
    center = np.asarray(ast.literal_eval(first["center"]), dtype=float)
    scale = np.asarray(ast.literal_eval(first["scale"]), dtype=float)

    args.out.mkdir(parents=True, exist_ok=True)
    cfg.to_csv(args.out / "settings_used.csv")

    trials = []
    labels = []

    for phase in PHASES:
        fig, axes = plt.subplots(
            2, 2, figsize=(12, 7), sharex=True,
            layout="constrained",
        )

        for seed in seeds:
            for idx, item in enumerate(paired_cases(profile, seed, phase)):
                case, clean, obs, start, end, onset, width, multiplier = item

                smoothed = filtered(obs, "wsg", size)
                times, block_features = features(smoothed)
                detector_scores = scores((block_features - center) / scale)

                labels.append({
                    "phase": phase,
                    "seed": seed,
                    "case": case,
                    "noise_multiplier": multiplier,
                    "transition_start_s": start,
                    "plateau_start_s": end,
                    "pause_start_s": onset if case == "pause" else None,
                    "resume_s": onset + width if case == "pause" else None,
                })

                # Always plot the FIRST seed, not the best-looking result.
                example = seed == args.seed_start

                if example:
                    trace = pd.DataFrame(
                        obs,
                        columns=[ch + "_value" for ch in CH],
                    )
                    trace.insert(0, "elapsed_seconds", np.arange(len(obs)))
                    trace["clean_ph"] = clean[:, 0]
                    trace["smoothed_ph"] = smoothed[:, 0]
                    trace.to_csv(
                        args.out / f"example_{phase}_{case}.csv",
                        index=False,
                    )

                    ax = axes.flat[idx]
                    ax.plot(
                        trace["elapsed_seconds"], obs[:, 0],
                        color="grey", lw=0.8, label="Observed",
                    )
                    ax.plot(
                        trace["elapsed_seconds"], clean[:, 0],
                        "k--", lw=1, label="Clean",
                    )

                    if start is not None:
                        ax.axvline(
                            start, color="purple", ls=":",
                            label="Transition start",
                        )
                        ax.axvline(
                            end, color="green", ls="--",
                            label="Plateau start",
                        )

                    if case == "pause":
                        ax.axvspan(
                            onset, onset + width,
                            color="orange", alpha=0.18,
                            label="Temporary pause",
                        )

                    ax.set(
                        title=case,
                        xlabel="Elapsed seconds",
                        ylabel="pH",
                    )

                for method in METHODS:
                    raw_alarms, original_k2 = decide(
                        times,
                        detector_scores[method],
                        cfg.loc[method, "threshold"],
                        block_features[:, 0],
                        phase,
                        profile["decline"],
                    )

                    # confirm() from run_confirmation.py was written for
                    # declining pH. Reflect ONLY its input for desorption.
                    # BOCPD still receives the actual, unreflected features.
                    ph_for_confirmation = (
                        block_features[:, 0]
                        if phase == "absorption"
                        else -block_features[:, 0]
                    )

                    for k in (2, 4):
                        switch = confirm(
                            times,
                            ph_for_confirmation,
                            raw_alarms,
                            profile["decline"],
                            (k, False),
                        )

                        # Verify the generalisation against the original
                        # phase-aware K2 rule on EVERY simulated trace.
                        if k == 2:
                            reconstructed = [] if switch is None else [switch]
                            if reconstructed != original_k2:
                                raise AssertionError(
                                    "K2 differs from the original pilot rule."
                                )

                        target = end is not None

                        # Generic change alarm: compare with transition START.
                        matched_raw = (
                            [r for r in raw_alarms if start <= r <= start + TOL]
                            if target else []
                        )

                        # Switching recommendation: compare with plateau START.
                        hit = (
                            target
                            and switch is not None
                            and end <= switch <= end + TOL
                        )

                        trials.append({
                            "seed": seed,
                            "phase": phase,
                            "case": case,
                            "method": method,
                            "k": k,
                            "target": int(target),
                            "raw_alarms": json.dumps(raw_alarms),
                            "raw_onset_hit": int(bool(matched_raw)),
                            "raw_onset_delay_s": (
                                matched_raw[0] - start
                                if matched_raw else np.nan
                            ),
                            "switch_s": switch,
                            "switched": int(switch is not None),
                            "hit": int(hit),
                            "premature": int(
                                target and switch is not None and switch < end
                            ),
                            "switch_delay_s": switch - end if hit else np.nan,
                        })

                        # Plot the multivariate method only for readability.
                        # CSVs still contain all three methods.
                        if example and method == "multi":
                            if k == 2:
                                ax.scatter(
                                    raw_alarms,
                                    obs[np.asarray(raw_alarms, dtype=int), 0],
                                    s=25, color="purple", label="Change alarm",
                                )
                            if switch is not None:
                                ax.scatter(
                                    switch, obs[switch, 0],
                                    s=60 if k == 2 else 30,
                                    marker="o" if k == 2 else "x",
                                    color="#0072b2" if k == 2 else "#d55e00",
                                    label=f"K{k} switch",
                                )

                if example:
                    ax.legend(fontsize=7, ncol=2)

        fig.suptitle(
            f"{phase.title()}-like pilot | WSG + multivariate BOCPD | "
            f"seed {args.seed_start}"
        )
        fig.savefig(args.out / f"examples_{phase}.png", dpi=180)
        plt.close(fig)

    pd.DataFrame(labels).to_csv(args.out / "event_labels.csv", index=False)

    results = pd.DataFrame(trials)
    results.to_csv(args.out / "trial_results.csv", index=False)

    summary = (
        results.groupby(["phase", "case", "method", "k"], as_index=False)
        .agg(
            n=("seed", "size"),
            targets=("target", "sum"),
            raw_hits=("raw_onset_hit", "sum"),
            hits=("hit", "sum"),
            switches=("switched", "sum"),
            premature=("premature", "sum"),
            median_raw_onset_delay_s=("raw_onset_delay_s", "median"),
            median_switch_delay_s=("switch_delay_s", "median"),
        )
    )

    denominator = summary["targets"].replace(0, np.nan)
    summary["raw_onset_recall"] = summary["raw_hits"] / denominator
    summary["switch_recall"] = summary["hits"] / denominator
    summary["misses"] = summary["targets"] - summary["hits"]
    summary["false_switch_fraction_no_target"] = (
        summary["switches"] / summary["n"]
    ).where(summary["targets"].eq(0))

    summary.to_csv(args.out / "summary.csv", index=False)

    # Paired comparison: did the same seed succeed for gradual vs abrupt?
    # One difference per independent seed, not one per second of data.
    paired = results[results["case"].isin(["abrupt", "gradual"])].pivot(
        index=["phase", "method", "k", "seed"],
        columns="case",
        values="hit",
    )
    difference = (paired["gradual"] - paired["abrupt"]).rename("difference")
    (
        difference.groupby(level=["phase", "method", "k"])
        .agg(["mean", "sem"])
        .to_csv(args.out / "paired_gradual_minus_abrupt_recall.csv")
    )

    print("\nMultivariate BOCPD results:")
    print(
        summary.loc[
            summary["method"].eq("multi"),
            [
                "phase", "case", "k", "hits", "targets", "premature",
                "median_switch_delay_s", "false_switch_fraction_no_target",
            ],
        ].to_string(index=False)
    )
    print(f"\nSaved to: {args.out.resolve()}")


if __name__ == "__main__":
    main()