from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import lag_correlation


def select_examples(statistics):
    """
    Select windows nearest the 10th, 50th and 90th percentiles
    of residual SD within each mode/gas group.

    These are diagnostic examples, not automatically approved
    noise-calibration windows.
    """
    selections = {}

    for _, group in statistics.groupby(
        ["segment_id", "mode", "gas_type"],
        dropna=False,
        sort=False,
    ):
        for label, quantile in (
            ("low", 0.10),
            ("middle", 0.50),
            ("high", 0.90),
        ):
            target = group["residual_sd"].quantile(quantile)

            distance = (group["residual_sd"] - target).abs()
            selected = group.loc[distance.idxmin()]
            window_id = selected["window_id"]

            selections.setdefault(window_id, []).append(label)

    return selections


def create_characterization_plots(
    statistics,
    details,
    output_dir,
    config,
):
    output_dir = Path(output_dir)
    plot_dir = output_dir / "plots"
    example_dir = output_dir / "examples"

    plot_dir.mkdir(parents=True, exist_ok=True)
    example_dir.mkdir(parents=True, exist_ok=True)

    if statistics.empty:
        print("No accepted windows: skipped diagnostic plots.")
        return

    segment_ids = statistics["segment_id"].drop_duplicates().tolist()

    # ------------------------------------------------------------
    # 1. Window-level comparison
    # ------------------------------------------------------------
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(13, 11),
        sharex=True,
        constrained_layout=True,
    )

    measurements = [
        ("slope_ph_per_minute", "Local slope\n(pH/min)"),
        ("residual_sd", "Residual SD\n(pH units)"),
        ("residual_lag1_correlation", "Residual lag-1\ncorrelation"),
        ("unchanged_step_fraction", "Unchanged-step\nfraction"),
    ]

    colors = plt.get_cmap("tab10")

    for position, segment_id in enumerate(segment_ids):
        group = statistics.loc[statistics["segment_id"].eq(segment_id)]

        # Deterministic horizontal spread; not statistical randomness.
        offsets = (
            np.linspace(-0.22, 0.22, len(group)) if len(group) > 1 else np.zeros(1)
        )

        x = position + offsets

        for ax, (column, ylabel) in zip(axes, measurements):
            values = group[column].to_numpy(dtype=float)
            valid = np.isfinite(values)

            ax.scatter(
                x[valid],
                values[valid],
                color=colors(position % 10),
                s=22,
                alpha=0.7,
            )

            if valid.any():
                median = float(np.median(values[valid]))

                ax.plot(
                    [position - 0.28, position + 0.28],
                    [median, median],
                    color="black",
                    linewidth=2,
                )

            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.2)

    axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[2].set_ylim(-1.05, 1.05)
    axes[3].set_ylim(-0.05, 1.05)

    axes[-1].set_xticks(
        range(len(segment_ids)),
        segment_ids,
    )
    axes[-1].set_xlabel("Retained segment")

    fig.suptitle(
        "pH characterization by analysis window\n"
        "Each point is one window; black bars show segment medians",
        fontsize=12,
    )

    fig.savefig(
        plot_dir / "window_characterization.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ------------------------------------------------------------
    # 2. Representative diagnostic examples
    # ------------------------------------------------------------
    selections = select_examples(statistics)
    selected_records = []

    for window_id, labels in selections.items():
        row = statistics.loc[statistics["window_id"].eq(window_id)].iloc[0]

        detail = details[window_id]
        residuals = detail["residual"].to_numpy(dtype=float)
        times = detail["elapsed_seconds"].to_numpy(dtype=float)

        figure_name = f"{window_id}.png"
        data_name = f"{window_id}.csv"

        detail.to_csv(
            example_dir / data_name,
            index=False,
        )

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(12, 7),
            constrained_layout=True,
        )

        # Original signal and fitted local line.
        ax = axes[0, 0]

        ax.plot(
            times,
            detail["ph_value"],
            marker=".",
            markersize=4,
            linewidth=0.8,
            label="Recorded pH",
        )

        ax.plot(
            times,
            detail["local_linear_trend"],
            color="darkorange",
            linewidth=2,
            label="Fitted local linear trend",
        )

        ax.set_ylabel("pH")
        ax.set_title("Observed signal and local trend")
        ax.legend(fontsize=8)

        # Residual time series.
        ax = axes[0, 1]

        ax.plot(
            times,
            residuals,
            marker=".",
            markersize=4,
            linewidth=0.8,
            color="teal",
        )

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Residual (pH units)")
        ax.set_title("Deviations from the fitted trend")

        # Residual distribution.
        ax = axes[1, 0]

        if np.std(residuals) <= config.numerical_tolerance:
            ax.text(
                0.5,
                0.5,
                "No residual variation above\nnumerical tolerance",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        else:
            bins = min(
                12,
                max(3, int(np.sqrt(len(residuals)))),
            )

            ax.hist(
                residuals,
                bins=bins,
                color="teal",
                alpha=0.75,
                edgecolor="white",
            )

        ax.set_xlabel("Residual (pH units)")
        ax.set_ylabel("Count")
        ax.set_title("Residual distribution")

        # Descriptive lag correlations.
        ax = axes[1, 1]

        max_lag = min(15, len(residuals) // 4)
        lags = np.arange(1, max_lag + 1)

        correlations = np.array(
            [
                lag_correlation(
                    residuals,
                    lag=int(lag),
                    tolerance=config.numerical_tolerance,
                )
                for lag in lags
            ]
        )

        valid = np.isfinite(correlations)

        if valid.any():
            ax.bar(
                lags[valid] * config.expected_dt_seconds,
                correlations[valid],
                width=0.7 * config.expected_dt_seconds,
                color="slateblue",
            )
        else:
            ax.text(
                0.5,
                0.5,
                "Correlation undefined:\ninsufficient variation",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("Lag (seconds)")
        ax.set_ylabel("Correlation")
        ax.set_title("Residual temporal dependence")

        for ax in axes[0]:
            ax.set_xlabel("Elapsed time within window (seconds)")

        for ax in axes.ravel():
            ax.grid(alpha=0.15)

        fig.suptitle(
            f"{window_id} | {row['mode']} | {row['gas_type']}\n"
            f"Example category: {', '.join(labels)} | "
            f"Slope: {row['slope_ph_per_minute']:.4f} pH/min | "
            f"Residual SD: {row['residual_sd']:.5f}",
            fontsize=11,
        )

        fig.savefig(
            plot_dir / figure_name,
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

        selected_records.append(
            {
                "window_id": window_id,
                "segment_id": row["segment_id"],
                "mode": row["mode"],
                "gas_type": row["gas_type"],
                "example_category": ", ".join(labels),
                "residual_sd": row["residual_sd"],
                "figure": figure_name,
                "data_file": data_name,
            }
        )

    pd.DataFrame(selected_records).to_csv(
        output_dir / "example_index.csv",
        index=False,
    )

    print(f"Diagnostic figures saved to: {plot_dir.resolve()}")
