import argparse
import ast
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_multichannel_pilot import (
    BLOCK,
    TOL,
    WARMUP,
    read_segment,
    characterize,
    calibrate,
    filtered,
    features,
    bocpd,
    decide,
)
from run_plateau_stress import paired_cases

# K = number of consecutive near-flat 30-second blocks.
# False/True = whether to cancel a pending candidate after renewed decline.
BASELINE = (2, False)
RULES = [(k, cancel) for k in (2, 3, 4) for cancel in (False, True)]

# Same inclusive candidate lifetime for all six rules.
EXPIRY_SECONDS = 120


def rule_name(rule):
    k, cancel = rule
    return f"K{k}_{'cancel' if cancel else 'keep'}"


def wilson_interval(k, n):
    """Pointwise 95% binomial interval, including when k=0 or k=n."""
    z = 1.95996398454
    p = k / n
    denominator = 1 + z**2 / n
    middle = (p + z**2 / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return max(0, middle - half), min(1, middle + half)


def confirm(times, ph_slopes, raw_alarms, decline, rule):
    """
    Turn a fixed sequence of raw BOCPD alarms into at most one switch.

    Only this function changes between candidate policies.
    It receives no simulated endpoint or plateau-duration label.
    """
    k, cancel = rule
    alarm_times = set(raw_alarms)

    opened = None
    expires = -1
    seen_flat = False

    for i, t in enumerate(times):
        if t < WARMUP:
            continue

        if t > expires:
            opened = None
            seen_flat = False

        # Same directional requirement as the original absorption rule:
        # the pH must previously have been declining.
        previous = ph_slopes[max(0, i - 5) : i]
        previously_declining = np.any(previous < -0.5 * decline)

        if t in alarm_times and previously_declining:
            opened = t
            expires = t + EXPIRY_SECONDS
            seen_flat = False

        if opened is None:
            continue

        # Optional cancellation: after seeing some flattening, withdraw
        # this pending candidate if a completed block shows renewed decline.
        # A later qualifying BOCPD alarm can open a new candidate.
        if cancel and t > opened and seen_flat and ph_slopes[i] < -0.5 * decline:
            opened = None
            expires = -1
            seen_flat = False
            continue

        seen_flat = seen_flat or abs(ph_slopes[i]) <= 0.30 * decline

        # Preserve the original trailing-history convention. The qualifying
        # blocks can include blocks completed before the raw alarm.
        if i + 1 >= k:
            recent = ph_slopes[i - k + 1 : i + 1]
            if np.all(np.abs(recent) <= 0.30 * decline):
                return int(t)

    return None


def run_split(seeds, profile, settings, rules):
    """
    Each seed generates a paired set:
        continuous decline, persistent flattening,
        temporary flattening for 30/60/90/120/180 seconds.

    Compute smoothing and BOCPD once per trace, then pass the SAME raw
    alarms and pH slopes to every confirmation rule.
    """
    rows = []

    for seed in seeds:
        for case, duration, onset, noise, _, observed in paired_cases(profile, seed):
            smoothed = filtered(observed, "wsg", settings["size"])
            times, slopes = features(smoothed)

            standardised = (slopes - settings["center"]) / settings["scale"]

            change_score = bocpd(standardised)

            # Reuse the original implementation for raw alarms and for an
            # independent check of our two-block comparator.
            raw_alarms, original_switches = decide(
                times,
                change_score,
                settings["threshold"],
                slopes[:, 0],
                "absorption",
                profile["decline"],
            )
            original_time = original_switches[0] if original_switches else None

            for rule in rules:
                switch_time = confirm(
                    times,
                    slopes[:, 0],
                    raw_alarms,
                    profile["decline"],
                    rule,
                )

                if rule == BASELINE and switch_time != original_time:
                    raise AssertionError(
                        "The two-block comparator did not reproduce "
                        "the original decision rule."
                    )

                # Only the persistent plateau is a desired switch.
                target = case == "permanent"
                hit = (
                    target
                    and switch_time is not None
                    and onset <= switch_time <= onset + TOL
                )

                if switch_time is None:
                    timing = "none"
                elif case == "decline":
                    timing = "during_decline"
                elif switch_time < onset:
                    timing = "early"
                elif case == "temporary":
                    timing = (
                        "on_plateau"
                        if switch_time <= onset + duration
                        else "after_resumption"
                    )
                else:
                    timing = "matched" if hit else "late"

                rows.append(
                    {
                        "seed": seed,
                        "rule": rule_name(rule),
                        "case": case,
                        "duration_s": duration,
                        "noise": noise,
                        "onset_s": onset if case != "decline" else np.nan,
                        "raw_alarms": str(raw_alarms),
                        "switch_s": switch_time,
                        "timing": timing,
                        "switched": int(switch_time is not None),
                        "hit": int(hit),
                        "delay_s": switch_time - onset if hit else np.nan,
                    }
                )

    return pd.DataFrame(rows)


def summarise(trials):
    """One performance row per confirmation rule."""
    rows = []

    for rule, group in trials.groupby("rule"):
        temporary = group[group["case"].eq("temporary")]
        persistent = group[group["case"].eq("permanent")]
        decline_only = group[group["case"].eq("decline")]

        # Durations share random draws within a seed. Average within seed
        # before calculating uncertainty across independent seeds.
        false_by_seed = temporary.groupby("seed")["switched"].mean()

        recall = persistent["hit"].mean()
        detected_delays = persistent["delay_s"].dropna()

        # Selection criterion only: a missed, early or late recommendation
        # gets the full tolerance penalty instead of an artificial zero delay.
        penalised_delay = persistent["delay_s"].fillna(TOL).mean()

        loss = false_by_seed.mean() + (1 - recall) + 0.25 * penalised_delay / TOL

        rows.append(
            {
                "rule": rule,
                "recall": recall,
                "hits": persistent["hit"].sum(),
                "n": len(persistent),
                "temp_false": false_by_seed.mean(),
                "temp_false_mcse": false_by_seed.sem(),
                "after_resumption": temporary["timing"].eq("after_resumption").mean(),
                "decline_false": decline_only["switched"].mean(),
                "early": persistent["timing"].eq("early").sum(),
                "median_hit_delay_s": (
                    detected_delays.median() if len(detected_delays) else np.nan
                ),
                "penalised_delay_s": penalised_delay,
                "loss": loss,
            }
        )

    return pd.DataFrame(rows)


def by_duration(trials):
    """Separate false switches before, during and after each pause."""
    rows = []
    temporary = trials[trials["case"].eq("temporary")]

    for (rule, duration), group in temporary.groupby(["rule", "duration_s"]):
        count = int(group["switched"].sum())
        n = len(group)
        low, high = wilson_interval(count, n)

        rows.append(
            {
                "rule": rule,
                "duration_s": duration,
                "n": n,
                "false_switches": count,
                "fraction": count / n,
                "ci_low": low,
                "ci_high": high,
                "early": group["timing"].eq("early").sum(),
                "on_plateau": group["timing"].eq("on_plateau").sum(),
                "after_resumption": group["timing"].eq("after_resumption").sum(),
            }
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment", required=True, type=Path)
    parser.add_argument("--pilot", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dev", type=int, default=100)
    parser.add_argument("--test", type=int, default=400)
    parser.add_argument("--dev-start", type=int, default=600000)
    parser.add_argument("--test-start", type=int, default=700000)
    args = parser.parse_args()

    dev_seeds = range(args.dev_start, args.dev_start + args.dev)
    test_seeds = range(args.test_start, args.test_start + args.test)

    if (
        min(args.dev, args.test) < 2
        or min(args.dev_start, args.test_start) < 0
        or set(dev_seeds).intersection(test_seeds)
    ):
        parser.error("Use at least two seeds and disjoint, nonnegative ranges.")

    original_test_seeds = set(
        pd.read_csv(args.pilot / "test_events.csv", usecols=["seed"])["seed"]
    )
    if original_test_seeds.intersection(set(dev_seeds) | set(test_seeds)):
        parser.error("Seed overlap with the original pilot test set.")

    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Choose a new or empty output directory.")

    if BLOCK != 30 or TOL != 180:
        parser.error("This protocol assumes 30 s blocks and 180 s matching tolerance.")

    # Verify S005 corresponds to the supplied pilot.
    _, real = read_segment(args.segment)
    stats = characterize(real)
    reference = pd.read_csv(args.pilot / "real_windows.csv")
    columns = ["slope_per_min", "residual_sd", "lag1", "unchanged"]

    def ordered(frame):
        return frame.sort_values(["channel", "window"])[columns].to_numpy()

    current, previous = ordered(stats), ordered(reference)
    if current.shape != previous.shape or not np.allclose(
        current, previous, equal_nan=True
    ):
        parser.error("Source segment does not match the pilot statistics.")

    profile = calibrate(real, stats)

    # Load exactly one frozen pipeline.
    table = pd.read_csv(args.pilot / "frozen_settings.csv")
    selected = table[table["family"].eq("wsg") & table["method"].eq("multi")]
    if len(selected) != 1:
        parser.error("Expected exactly one WSG + multivariate BOCPD setting.")

    row = selected.iloc[0]
    settings = {
        "size": int(row["size"]),
        "threshold": float(row["threshold"]),
        "center": np.asarray(ast.literal_eval(row["center"])),
        "scale": np.asarray(ast.literal_eval(row["scale"])),
    }

    args.out.mkdir(parents=True, exist_ok=True)

    # STEP 1: Compare all six policies on development seeds.
    print("Development: comparing six confirmation rules...", flush=True)
    development = run_split(dev_seeds, profile, settings, RULES)
    development.to_csv(args.out / "dev_trials.csv", index=False)

    ranking = summarise(development)
    tie_order = {rule_name(rule): i for i, rule in enumerate(RULES)}
    ranking["tie_order"] = ranking["rule"].map(tie_order)
    ranking = ranking.sort_values(["loss", "tie_order"]).drop(columns="tie_order")
    ranking.to_csv(args.out / "dev_selection.csv", index=False)

    winner = next(rule for rule in RULES if rule_name(rule) == ranking.iloc[0]["rule"])

    # STEP 2: Lock the winner before generating the held-out test traces.
    # Also evaluate the opposite cancellation setting at the same K,
    # so cancellation can be assessed without changing duration.
    test_rules = list(
        dict.fromkeys(
            [
                BASELINE,
                winner,
                (winner[0], not winner[1]),
            ]
        )
    )

    lock = {
        "baseline": rule_name(BASELINE),
        "selected": rule_name(winner),
        "tested": [rule_name(rule) for rule in test_rules],
        "development_seed_range": [dev_seeds.start, dev_seeds.stop - 1],
        "test_seed_range": [test_seeds.start, test_seeds.stop - 1],
        "candidate_expiry_s": EXPIRY_SECONDS,
        "loss": "temp_false + (1-recall) + 0.25*mean_penalised_delay/180",
        "source_sha256": hashlib.sha256(args.segment.read_bytes()).hexdigest(),
        "filter": "wsg",
        "detector": "multi",
        "size": settings["size"],
        "threshold": settings["threshold"],
        "center": settings["center"].tolist(),
        "scale": settings["scale"].tolist(),
    }

    (args.out / "locked_rule.json").write_text(
        json.dumps(lock, indent=2), encoding="utf-8"
    )

    # STEP 3: Evaluate the locked comparison on fresh seeds.
    print("Held-out evaluation:", lock["tested"], flush=True)
    heldout = run_split(test_seeds, profile, settings, test_rules)
    heldout.to_csv(args.out / "test_trials.csv", index=False)

    summary = summarise(heldout)
    summary.to_csv(args.out / "test_summary.csv", index=False)

    durations = by_duration(heldout)
    durations.to_csv(args.out / "test_by_duration.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for rule, group in durations.groupby("rule"):
        group = group.sort_values("duration_s")
        (line,) = ax.plot(group["duration_s"], group["fraction"], "o-", label=rule)
        ax.fill_between(
            group["duration_s"].to_numpy(),
            group["ci_low"].to_numpy(),
            group["ci_high"].to_numpy(),
            color=line.get_color(),
            alpha=0.12,
        )

    ax.set(
        xlabel="Temporary plateau duration (seconds)",
        ylabel="Fraction of traces with a false switch",
        title="Held-out comparison of confirmation rules",
        ylim=(-0.03, 1.03),
    )
    ax.set_xticks(sorted(durations["duration_s"].unique()))
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(args.out / "false_switch_by_duration.png", dpi=180)
    plt.close(fig)

    print("\nSelected on development:", rule_name(winner))
    print(summary.round(4).to_string(index=False))
    print("\nSaved to:", args.out.resolve())


if __name__ == "__main__":
    main()
