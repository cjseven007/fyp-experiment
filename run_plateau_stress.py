import argparse
import ast
from pathlib import Path
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import lfilter

from run_multichannel_pilot import (
    GRID,
    TOL,
    read_segment,
    characterize,
    calibrate,
    filtered,
    features,
    scores,
    decide,
)

DURATIONS = (30, 60, 90, 120, 180)
METHODS = ("slope", "uni", "multi")
COLORS = ("#555555", "#0072b2", "#d55e00")


# ------------------------------------------------------------
# 1. Generate paired persistent and temporary plateaus
# ------------------------------------------------------------


def paired_cases(profile, seed):
    """
    Generate seven absorption-like traces from one seed:

        1 continuous decline
        1 persistent plateau
        5 temporary plateaus

    The traces share their initial pH, decline rate, temperature,
    pressure, fluctuations and rounding. Only plateau duration changes.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(600)  # 10 minutes at 1 Hz

    # Simulation choices; not measured reactor endpoint times.
    onset = int(rng.integers(245, 321))
    rate = -profile["decline"] * rng.uniform(0.75, 1.25) / 60
    initial_ph = profile["start"][0] + rng.uniform(-0.03, 0.03)

    # Same type of operating context as the original pilot.
    temperature = (
        profile["start"][1]
        + profile["heat"] * rng.uniform(0.6, 1.15) * t / 60
        + 0.05 * profile["span"][1] * np.sin(2 * np.pi * t / rng.uniform(100, 220))
    )

    pressure = profile["pressure"] + 0.12 * profile["span"][2] * np.sin(
        2 * np.pi * t / rng.uniform(70, 170) + rng.uniform(0, 2 * np.pi)
    )

    # Generate fluctuations ONCE per seed, then reuse them in all cases.
    multiplier = float(rng.choice([0.0, 0.5, 1.0]))
    noise = np.empty((len(t), 3))

    for j, (sd, rho) in enumerate(zip(profile["sd"], profile["rho"])):
        sigma = multiplier * sd

        innovations = rng.normal(size=len(t)) * sigma * np.sqrt(1 - rho * rho)
        innovations[0] = rng.normal() * sigma
        noise[:, j] = lfilter([1.0], [1.0, -rho], innovations)

    cases = [
        ("decline", 0),
        ("permanent", None),
    ] + [("temporary", duration) for duration in DURATIONS]

    for case, duration in cases:

        # "Active time" is the number of seconds during which
        # the underlying clean pH has been declining.
        if case == "decline":
            active_time = t

        elif case == "permanent":
            active_time = np.minimum(t, onset)

        else:
            # Before onset: normal decline.
            # During plateau: active time stops increasing.
            # After plateau: decline resumes from the plateau level.
            active_time = np.minimum(t, onset) + np.maximum(t - onset - duration, 0)

        clean_ph = initial_ph + rate * active_time

        clean = np.column_stack(
            [
                clean_ph,
                temperature,
                pressure,
            ]
        )

        observed = np.round((clean + noise) / GRID) * GRID

        yield case, duration, onset, multiplier, clean, observed


# ------------------------------------------------------------
# 2. Uncertainty for a proportion
# ------------------------------------------------------------


def wilson_interval(k, n):
    """95% Wilson binomial interval; works for 0/n and n/n."""
    z = NormalDist().inv_cdf(0.975)
    p = k / n
    denominator = 1 + z**2 / n

    middle = (p + z**2 / (2 * n)) / denominator
    half_width = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator

    return max(0.0, middle - half_width), min(1.0, middle + half_width)


# ------------------------------------------------------------
# 3. Run the frozen detectors and record when switches occur
# ------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Test false switching during temporary pH plateaus."
    )
    parser.add_argument("--segment", required=True, type=Path)
    parser.add_argument("--pilot", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--family",
        choices=["raw", "ema", "wsg"],
        default="wsg",
    )
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=500000)
    args = parser.parse_args()

    if args.reps < 1 or args.seed_start < 0:
        parser.error("Use positive repetitions and a nonnegative starting seed.")

    seeds = range(args.seed_start, args.seed_start + args.reps)

    old_test_seeds = pd.read_csv(
        args.pilot / "test_events.csv",
        usecols=["seed"],
    )["seed"]

    if set(seeds).intersection(old_test_seeds):
        parser.error("Choose seeds outside the original pilot test range.")

    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Choose a new or empty output directory.")

    # Verify that the source corresponds to the pilot being reused.
    _, real = read_segment(args.segment)
    real_stats = characterize(real)
    original_stats = pd.read_csv(args.pilot / "real_windows.csv")

    columns = ["slope_per_min", "residual_sd", "lag1", "unchanged"]

    def ordered(frame):
        return frame.sort_values(["window", "channel"])[columns].to_numpy()

    current = ordered(real_stats)
    original = ordered(original_stats)

    if current.shape != original.shape or not np.allclose(
        current, original, equal_nan=True
    ):
        raise ValueError(
            "The segment statistics do not match the selected pilot folder."
        )

    profile = calibrate(real, real_stats)

    # Reuse the exact filter, feature scaling and alarm thresholds
    # selected in the previous development experiment.
    settings = pd.read_csv(args.pilot / "frozen_settings.csv")
    settings = (
        settings.loc[settings["family"].eq(args.family)]
        .set_index("method")
        .loc[list(METHODS)]
    )

    row = settings.iloc[0]
    size = int(row["size"])
    center = np.array(ast.literal_eval(row["center"]), dtype=float)
    scale = np.array(ast.literal_eval(row["scale"]), dtype=float)

    args.out.mkdir(parents=True, exist_ok=True)

    trial_rows = []
    examples = []

    for seed in seeds:
        permanent_observed = None

        for case, duration, onset, multiplier, clean, observed in paired_cases(
            profile, seed
        ):
            if case == "permanent":
                permanent_observed = observed.copy()

            if case == "temporary":
                end = onset + duration

                # A temporary plateau must be indistinguishable from
                # the persistent one until decline actually resumes.
                assert np.array_equal(
                    observed[: end + 1],
                    permanent_observed[: end + 1],
                )

                # Temperature and pressure must remain identical throughout.
                assert np.array_equal(
                    observed[:, 1:],
                    permanent_observed[:, 1:],
                )

            smoothed = filtered(observed, args.family, size)
            times, block_features = features(smoothed)
            all_scores = scores((block_features - center) / scale)

            for method in METHODS:
                changes, switches = decide(
                    times,
                    all_scores[method],
                    settings.loc[method, "threshold"],
                    block_features[:, 0],
                    "absorption",
                    profile["decline"],
                )

                first_switch = switches[0] if switches else np.nan

                # Only the persistent plateau is a target.
                # Use the same matching tolerance as the original pilot.
                hit = bool(case == "permanent" and onset <= first_switch <= onset + TOL)

                if not switches:
                    timing = "none"
                elif case == "decline":
                    timing = "during_decline"
                elif first_switch < onset:
                    timing = "before_onset"
                elif case == "temporary":
                    timing = (
                        "on_plateau"
                        if first_switch <= onset + duration
                        else "after_resumption"
                    )
                else:
                    timing = "matched" if hit else "late"

                trial_rows.append(
                    {
                        "seed": seed,
                        "case": case,
                        "duration_s": duration,
                        "onset_s": onset if case != "decline" else np.nan,
                        "multiplier": multiplier,
                        "family": args.family,
                        "method": method,
                        "change_alarms_s": str(changes),
                        "first_switch_s": first_switch,
                        "timing": timing,
                        "any_switch": int(bool(switches)),
                        "hit": int(hit),
                        "delay_s": first_switch - onset if hit else np.nan,
                    }
                )

                # Fix the example seed in advance rather than choosing
                # the most convincing-looking result after evaluation.
                if (
                    seed == args.seed_start
                    and method == "multi"
                    and (case == "permanent" or duration in (60, 120))
                ):
                    examples.append(
                        (
                            case,
                            duration,
                            onset,
                            clean[:, 0],
                            observed[:, 0],
                            changes,
                            switches,
                        )
                    )

    trials = pd.DataFrame(trial_rows)
    trials.to_csv(args.out / "trials.csv", index=False)

    # --------------------------------------------------------
    # 4. Summarise results
    # --------------------------------------------------------
    summary_rows = []

    for (case, duration, method), group in trials.groupby(
        ["case", "duration_s", "method"],
        dropna=False,
    ):
        is_target = case == "permanent"

        # Persistent case: proportion correctly matched.
        # Other cases: proportion receiving any false switch.
        count = int(group["hit" if is_target else "any_switch"].sum())
        n = len(group)
        low, high = wilson_interval(count, n)

        delays = group["delay_s"].dropna()

        summary_rows.append(
            {
                "case": case,
                "duration_s": duration,
                "method": method,
                "n": n,
                "count": count,
                "metric": "recall" if is_target else "false_switch_fraction",
                "fraction": count / n,
                "ci_low": low,
                "ci_high": high,
                "before_onset": int(group["timing"].eq("before_onset").sum()),
                "on_plateau": int(group["timing"].eq("on_plateau").sum()),
                "after_resumption": int(group["timing"].eq("after_resumption").sum()),
                "median_hit_delay_s": (delays.median() if len(delays) else np.nan),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out / "summary.csv", index=False)

    # --------------------------------------------------------
    # 5. Plot false switching against plateau duration
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")

    for method, color in zip(METHODS, COLORS):
        part = summary.loc[
            summary["case"].eq("temporary") & summary["method"].eq(method)
        ].sort_values("duration_s")

        x = part["duration_s"].to_numpy()
        y = part["fraction"].to_numpy()

        ax.plot(x, y, "o-", color=color, label=method)
        ax.fill_between(
            x,
            part["ci_low"].to_numpy(),
            part["ci_high"].to_numpy(),
            color=color,
            alpha=0.10,
        )

    ax.set(
        xlabel="Temporary plateau duration (seconds)",
        ylabel="Fraction of traces with a false switch",
        title=f"{args.family.upper()}: temporary pH plateau stress test",
        ylim=(-0.03, 1.03),
    )
    ax.set_xticks(DURATIONS)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(args.out / "false_switch_by_duration.png", dpi=180)
    plt.close(fig)

    # --------------------------------------------------------
    # 6. Plot paired examples on the same time axis
    # --------------------------------------------------------
    fig, axes = plt.subplots(
        len(examples),
        1,
        sharex=True,
        figsize=(10, 7),
        layout="constrained",
    )

    for ax, example in zip(axes, examples):
        case, duration, onset, clean_ph, observed_ph, changes, switches = example
        t = np.arange(len(observed_ph))

        ax.plot(t, observed_ph, color="grey", lw=0.8, label="Observed synthetic pH")
        ax.plot(t, clean_ph, color="black", ls="--", lw=1, label="Clean pH")

        plateau_end = len(t) - 1 if duration is None else onset + duration
        ax.axvspan(onset, plateau_end, color="gold", alpha=0.18)

        ax.scatter(
            changes,
            observed_ph[np.asarray(changes, dtype=int)],
            marker="o",
            color="#0072b2",
            label="Change alarm",
        )
        ax.scatter(
            switches,
            observed_ph[np.asarray(switches, dtype=int)],
            marker="D",
            color="#d55e00",
            label="Switch recommendation",
        )

        ax.set_ylabel("pH")
        ax.set_title(
            "Persistent plateau to end of trace"
            if duration is None
            else f"Temporary plateau: {duration} seconds"
        )

    axes[0].legend(fontsize=8, ncol=4)
    axes[-1].set_xlabel("Elapsed seconds")
    fig.suptitle(
        f"Paired traces with identical initial observations | "
        f"{args.family.upper()} + multivariate BOCPD"
    )
    fig.savefig(args.out / "paired_examples.png", dpi=180)
    plt.close(fig)

    print(
        summary[
            ["case", "duration_s", "method", "metric", "count", "n", "fraction"]
        ].to_string(index=False)
    )
    print(f"\nSaved to: {args.out.resolve()}")


if __name__ == "__main__":
    main()
