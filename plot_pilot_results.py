import argparse
import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_multichannel_pilot import filtered, features, scores, decide

CHANNELS = ["ph", "temperature", "pressure"]
UNITS = {"ph": "pH", "temperature": "°C", "pressure": "barg"}

FAMILIES = ["raw", "ema", "wsg"]
METHODS = ["slope", "uni", "multi"]
CASES = ["A_decline_or_rise", "B_flattening", "C_pressure_pulse"]

FAMILY_COLORS = ["#59636f", "#0072b2", "#d55e00"]
BLUE = "#0072b2"
ORANGE = "#d55e00"


def save(fig, folder, name):
    """Save and close a figure without changing any input CSV."""
    fig.savefig(folder / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------
# 00. Show the three pilot signals AND the decisions they cause
# ------------------------------------------------------------


def plot_pilot_decisions(root, out):
    # Use WSG for this example so that the figure stays readable.
    # Later figures compare all three smoothing families.
    settings = pd.read_csv(root / "frozen_settings.csv")
    settings = settings.loc[settings["family"].eq("wsg")].set_index("method")

    windows = pd.read_csv(root / "real_windows.csv")
    negative_ph = windows.loc[
        windows["channel"].eq("ph") & windows["slope_per_min"].lt(-1e-8),
        "slope_per_min",
    ]
    decline = -negative_ph.median()

    # In the benchmark, all detectors within one smoothing family
    # use the same filter size and feature standardisation.
    row = settings.loc["multi"]
    size = int(row["size"])
    center = np.asarray(ast.literal_eval(row["center"]), dtype=float)
    scale = np.asarray(ast.literal_eval(row["scale"]), dtype=float)

    labels = pd.read_csv(root / "pilot_labels.csv").set_index("case")

    fig, axes = plt.subplots(
        4,
        3,
        figsize=(14, 8),
        sharex=True,
        sharey="row",
        layout="constrained",
    )

    titles = [
        "A: continued decline",
        "B: pH flattens",
        "C: pressure pulse",
    ]

    for k, case in enumerate(CASES):
        d = pd.read_csv(root / f"{case}.csv")
        y = d[[f"{ch}_value" for ch in CHANNELS]].to_numpy()
        minutes = d["elapsed_seconds"].to_numpy() / 60

        # Reapply frozen settings. No fitting or threshold tuning here.
        smoothed = filtered(y, "wsg", size)
        times, block_features = features(smoothed)
        all_scores = scores((block_features - center) / scale)

        # Top three panels in this column: recorded and smoothed signals.
        for j, ch in enumerate(CHANNELS):
            ax = axes[j, k]
            ax.plot(minutes, y[:, j], lw=0.7, color="grey")
            ax.plot(minutes, smoothed[:, j], lw=1.1, color=BLUE)

            if k == 0:
                ax.set_ylabel(f"{ch} ({UNITS[ch]})")

        # Bottom panel: actual alarm times for each detection method.
        for i, method in enumerate(METHODS):
            raw_alarms, switches = decide(
                times,
                all_scores[method],
                settings.loc[method, "threshold"],
                block_features[:, 0],  # pH slope for confirmation
                "absorption",  # the saved example trio is absorption-like
                decline,
            )

            for events, marker, color, offset in [
                (raw_alarms, "o", BLUE, -0.11),
                (switches, "D", ORANGE, 0.11),
            ]:
                axes[3, k].scatter(
                    np.asarray(events) / 60,
                    np.full(len(events), i + offset),
                    marker=marker,
                    color=color,
                    s=30,
                )

        axes[3, k].set_yticks(range(3), METHODS)
        axes[3, k].set_ylim(2.5, -0.5)
        axes[3, k].set_xlabel("Elapsed minutes")
        axes[0, k].set_title(titles[k])

        # Truth labels are added ONLY to the finished plot.
        # They were not passed to the detector.
        lab = labels.loc[case]
        for ax in axes[:, k]:
            if pd.notna(lab["target_s"]):
                ax.axvline(lab["target_s"] / 60, color="green", linestyle="--")

            if pd.notna(lab["nuisance_start_s"]):
                ax.axvspan(
                    lab["nuisance_start_s"] / 60,
                    lab["nuisance_end_s"] / 60,
                    color="orange",
                    alpha=0.17,
                )

            ax.grid(alpha=0.15)

    fig.suptitle(
        "One absorption example per case: grey = observed, blue line = causal WSG\n"
        "Blue circle = change alarm; orange diamond = switch recommendation; "
        "green line = target; shading = disturbance"
    )
    save(fig, out, "00_pilot_decisions")


# ------------------------------------------------------------
# 01. Compare synthetic recording characteristics with S005
# ------------------------------------------------------------


def plot_realism(root, out):
    df = pd.read_csv(root / "pilot_vs_real.csv")
    cases = ["real_S005"] + CASES

    metrics = [
        ("slope_per_min", "Slope / min"),
        ("residual_sd", "Residual SD"),
        ("lag1", "Lag-1 correlation"),
        ("unchanged", "Unchanged fraction"),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(13, 8), layout="constrained")

    for i, ch in enumerate(CHANNELS):
        part = df.loc[df["channel"].eq(ch)].set_index("case").reindex(cases)

        for j, (column, title) in enumerate(metrics):
            ax = axes[i, j]
            values = part[column].to_numpy(dtype=float)

            ax.scatter(
                range(4),
                values,
                c=["black", BLUE, "#009e73", ORANGE],
            )

            if np.isfinite(values[0]):
                ax.axhline(values[0], ls="--", color="grey", lw=1)

            ax.set_xticks(range(4), ["Real", "A", "B", "C"])

            if i == 0:
                ax.set_title(title)
            if j == 0:
                ax.set_ylabel(f"{ch}\n({UNITS[ch]}/min)")
            elif j == 1:
                ax.set_ylabel(UNITS[ch])
            elif j == 2:
                ax.set_ylim(-1.05, 1.05)
            elif j == 3:
                ax.set_ylim(-0.05, 1.05)

            ax.grid(axis="y", alpha=0.2)

    fig.suptitle(
        "Window medians: S005 versus one paired synthetic example\n"
        "Dashed line = S005 reference; matching it is not required in every scenario"
    )
    save(fig, out, "01_realism")


# ------------------------------------------------------------
# 02–03. Show filter tuning and held-out reconstruction error
# ------------------------------------------------------------


def plot_smoothing(root, out):
    tune = pd.read_csv(root / "filter_tuning.csv")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharey=True, layout="constrained")

    for ax, family, color in zip(axes, FAMILIES, FAMILY_COLORS):
        g = tune.loc[tune["family"].eq(family)].sort_values("size")
        best = g.loc[g["loss"].idxmin()]

        ax.plot(g["size"], g["loss"], "o-", color=color)
        ax.scatter(
            best["size"],
            best["loss"],
            s=110,
            facecolors="none",
            edgecolors="black",
        )
        ax.set_title(family.upper())
        ax.set_xlabel("Candidate size (samples)")
        ax.set_xticks(g["size"])
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Mean scaled RMSE")
    fig.suptitle("Development: ring = selected setting within each family")
    save(fig, out, "02_filter_selection")

    test = pd.read_csv(root / "test_rmse.csv")

    for phase, g in test.groupby("phase"):
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), layout="constrained")

        for ax, ch in zip(axes, CHANNELS):
            # A/B/C deliberately share random draws.
            # Average within seed rather than treating them as independent.
            values = g.groupby(["seed", "family"])[ch].mean().unstack()

            ax.boxplot([values[f].dropna() for f in FAMILIES])
            ax.set_xticks([1, 2, 3], ["Raw", "EMA", "WSG"])
            ax.set_title(ch)
            ax.set_ylabel(f"RMSE ({UNITS[ch]})")
            ax.grid(axis="y", alpha=0.2)

        fig.suptitle(
            f"Held-out {phase}: each observation is one seed, averaged over A/B/C"
        )
        save(fig, out, f"03_test_rmse_{phase}")


# ------------------------------------------------------------
# 04. Show detection success AND the cost in false alarms
# ------------------------------------------------------------


def plot_performance(root, out):
    df = pd.read_csv(root / "test_summary.csv")

    for (phase, layer), g in df.groupby(["phase", "layer"]):
        g = g.set_index(["family", "method", "case"])
        names, rows = [], []

        for family in FAMILIES:
            for method in METHODS:
                a, b, c = [g.loc[(family, method, case)] for case in CASES]

                names.append(f"{family.upper()} / {method}")
                rows.append(
                    [
                        b["recall"],
                        b["median_detected_delay_s"],
                        a["false_per_trace"],
                        c["false_per_trace"],
                    ]
                )

        values = np.asarray(rows, dtype=float)
        titles = [
            "B: recall",
            "B: delay (s)*",
            "A: false / trace",
            "C: false / trace",
        ]

        fig, axes = plt.subplots(1, 4, figsize=(12, 5.2), layout="constrained")

        for j, ax in enumerate(axes):
            column = values[:, j]
            valid = np.isfinite(column)

            ax.barh(
                np.arange(len(names))[valid],
                column[valid],
                color=np.repeat(FAMILY_COLORS, 3)[valid],
            )
            ax.set_yticks(
                range(len(names)),
                names if j == 0 else [""] * len(names),
            )
            ax.set_ylim(len(names) - 0.5, -0.5)

            upper = (
                1.0
                if j == 0
                else max(float(column[valid].max()) if valid.any() else 0, 0.1)
            )
            ax.set_xlim(0, upper * 1.30)

            for i, value in enumerate(column):
                if not valid[i]:
                    label = "N/A"
                elif j == 1:
                    label = f"{value:.0f}"
                else:
                    label = f"{value:.2f}"

                ax.text(
                    (value if valid[i] else 0) + 0.015 * upper,
                    i,
                    label,
                    va="center",
                    fontsize=8,
                )

            ax.set_title(titles[j], fontsize=10)
            ax.grid(axis="x", alpha=0.2)

        n = int(g["n"].iloc[0])
        fig.suptitle(
            f"{phase} | {layer} layer | {n} test seeds per case\n"
            "*Median delay among hits only. Read together with recall."
        )
        save(fig, out, f"04_performance_{phase}_{layer}")


# ------------------------------------------------------------
# 05. Place real alarms next to real S005 slope behaviour
# ------------------------------------------------------------


def plot_real_replay(root, out):
    windows = pd.read_csv(root / "real_windows.csv", parse_dates=["window_start"])
    alarms = pd.read_csv(root / "real_replay.csv")
    origin = windows["window_start"].min()

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(11, 9),
        sharex=True,
        height_ratios=[1, 1, 1, 2.7],
        layout="constrained",
    )

    # A horizontal line spans the minute used to calculate that slope.
    # It is an offline description, not an alarm available at minute start.
    for ax, ch in zip(axes[:3], CHANNELS):
        w = windows.loc[windows["channel"].eq(ch)].sort_values("window_start")
        start = (w["window_start"] - origin).dt.total_seconds() / 60

        ax.hlines(w["slope_per_min"], start, start + 1, lw=3, color=BLUE)
        ax.axhline(0, ls=":", color="grey")
        ax.set_ylabel(f"{ch}\n{UNITS[ch]}/min")
        ax.grid(axis="x", alpha=0.2)

    ax = axes[-1]
    names = []
    combinations = [(f, m) for f in FAMILIES for m in METHODS]

    for i, (family, method) in enumerate(combinations):
        names.append(f"{family.upper()} / {method}")

        for layer, marker, color, offset in [
            ("change", "o", BLUE, -0.13),
            ("switch", "D", ORANGE, 0.13),
        ]:
            match = alarms.loc[
                alarms["family"].eq(family)
                & alarms["method"].eq(method)
                & alarms["layer"].eq(layer)
            ]

            # The existing runner stores alarm lists as text, e.g. "[239, 539]".
            events = (
                np.asarray(ast.literal_eval(match.iloc[0]["alarms_s"]), dtype=float)
                / 60
            )

            ax.scatter(
                events,
                np.full(len(events), i + offset),
                marker=marker,
                color=color,
                s=28,
                label=layer if i == 0 else None,
            )

    ax.set_yticks(range(len(names)), names)
    ax.set_ylim(len(names) - 0.4, -0.6)
    ax.set_xlabel(f"Minutes since {origin:%H:%M:%S}")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid(alpha=0.18)

    fig.suptitle(
        "S005 replay: measured 60-second slopes and online alarm times\n"
        "No validated saturation time; a blank row means no alarms"
    )
    save(fig, out, "05_real_replay")


def main():
    parser = argparse.ArgumentParser(
        description="Plot results from run_multichannel_pilot.py."
    )
    parser.add_argument(
        "--results",
        required=True,
        type=Path,
        help="Existing multichannel pilot output directory.",
    )
    args = parser.parse_args()

    root = args.results
    if not (root / "frozen_settings.csv").is_file():
        parser.error(
            "Pass the multichannel pilot output folder containing "
            "frozen_settings.csv, not the prepared segment folder."
        )

    out = root / "result_plots"
    out.mkdir(parents=True, exist_ok=True)

    plot_pilot_decisions(root, out)
    plot_realism(root, out)
    plot_smoothing(root, out)
    plot_performance(root, out)
    plot_real_replay(root, out)

    print(f"Figures saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
