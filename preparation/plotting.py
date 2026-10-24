from pathlib import Path
from math import ceil

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

PLOT_ORDER = ["co2", "pressure", "temperature", "ph"]

LABELS = {
    "co2": "CO₂ (ppm)",
    "pressure": "Pressure (barg)",
    "temperature": "Temperature (°C)",
    "ph": "pH",
}

STATUS_STYLE = {
    "excluded_quality": {
        "label": "Excluded by quality screening",
        "color": "#C44E52",
    },
    "eligible_but_too_short": {
        "label": "Eligible but too short",
        "color": "#D99A24",
    },
    "retained": {
        "label": "Retained",
        "color": "#238B57",
    },
}


def format_time_axis(ax):
    locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def plot_without_bridging(
    ax,
    part,
    value_column,
    max_gap_seconds,
    color,
    label=None,
    linewidth=1.0,
):
    """Draw observed points without connecting across long time gaps."""
    values = pd.to_numeric(part[value_column], errors="coerce")
    values = values.where(np.isfinite(values))

    dt = part["timestamp"].diff().dt.total_seconds()

    # A gap/reversal begins a separate drawn line.
    groups = (dt.isna() | dt.le(0) | dt.gt(max_gap_seconds)).cumsum()

    first = True

    for _, piece in part.groupby(groups, sort=False):
        ax.plot(
            piece["timestamp"],
            values.loc[piece.index],
            color=color,
            linewidth=linewidth,
            label=label if first else None,
        )
        first = False

    # Preserve isolated observations too.
    ax.scatter(
        part["timestamp"],
        values,
        color=color,
        s=4,
        zorder=3,
    )


def create_preparation_plots(
    prepared,
    manifest,
    output_dir,
    channels=("ph",),
    max_gap_seconds=3.0,
):
    output_dir = Path(output_dir)
    plot_dir = output_dir / "plots"
    detail_dir = plot_dir / "retained_segments"

    detail_dir.mkdir(parents=True, exist_ok=True)

    df = prepared.copy().reset_index(drop=True)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    channels = [channel for channel in PLOT_ORDER if channel in channels]

    if not channels:
        raise ValueError("No supported channels selected.")

    required = [
        "preparation_status",
        "segment_id",
        *[f"{channel}_value" for channel in channels],
    ]

    missing = [column for column in required if column not in df]
    if missing:
        raise ValueError(f"Missing preparation columns: {missing}")

    unknown = set(df["preparation_status"].dropna()) - set(STATUS_STYLE)

    if unknown or df["preparation_status"].isna().any():
        raise ValueError(f"Unexpected or missing preparation status: {unknown}")

    # Percentages refer to rows, not elapsed recording time.
    counts = df["preparation_status"].value_counts().reindex(STATUS_STYLE, fill_value=0)

    summary = counts.rename("rows").rename_axis("status").reset_index()
    summary["percent_rows"] = 100.0 * summary["rows"] / len(df) if len(df) else 0.0

    summary.to_csv(
        output_dir / "preparation_status_counts.csv",
        index=False,
    )

    bad_time_count = int(df["timestamp"].isna().sum())

    visible = df.loc[df["timestamp"].notna()].copy()

    if visible.empty:
        print("No valid timestamps: saved counts, skipped figures.")
        return

    segment_ids = (
        visible.loc[
            visible["preparation_status"].eq("retained"),
            "segment_id",
        ]
        .dropna()
        .unique()
        .tolist()
    )

    if len(manifest) and "segment_id" in manifest.columns:
        metadata = manifest.set_index("segment_id").to_dict("index")
    else:
        metadata = {}

    def segment_title(segment_id, part):
        info = metadata.get(segment_id, {})

        mode = info.get(
            "mode",
            part["Mode"].iloc[0] if "Mode" in part else "Unknown",
        )

        gas = info.get(
            "gas_type",
            part["gas_type"].iloc[0] if "gas_type" in part else "Unknown",
        )

        duration = (
            part["timestamp"].iloc[-1] - part["timestamp"].iloc[0]
        ).total_seconds()

        return (
            f"{segment_id} | {mode} | {gas}\n"
            f"{len(part):,} readings | {duration / 60:.2f} min"
        )

    # ------------------------------------------------------------
    # 1. Original timeline + retained values + preparation mask
    # ------------------------------------------------------------
    fig, axes = plt.subplots(
        len(channels) + 1,
        1,
        figsize=(14, 3 * len(channels) + 2.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3] * len(channels) + [1.4]},
        constrained_layout=True,
        squeeze=False,
    )

    axes = axes[:, 0]

    for ax, channel in zip(axes[:-1], channels):
        column = f"{channel}_value"

        plot_without_bridging(
            ax,
            visible,
            column,
            max_gap_seconds,
            color="#B8B8B8",
            linewidth=0.7,
        )

        # Mark exclusions on the actual observed values.
        for status in (
            "excluded_quality",
            "eligible_but_too_short",
        ):
            mask = visible["preparation_status"].eq(status)
            values = pd.to_numeric(visible.loc[mask, column], errors="coerce")
            finite = np.isfinite(values)

            ax.scatter(
                visible.loc[mask, "timestamp"].loc[finite],
                values.loc[finite],
                color=STATUS_STYLE[status]["color"],
                s=9,
                zorder=4,
            )

        # Each retained segment is drawn separately.
        for segment_id in segment_ids:
            part = visible.loc[visible["segment_id"].eq(segment_id)]

            plot_without_bridging(
                ax,
                part,
                column,
                max_gap_seconds,
                color=STATUS_STYLE["retained"]["color"],
                linewidth=1.1,
            )

        ax.set_ylabel(LABELS[channel])
        ax.grid(alpha=0.2)

    mask_ax = axes[-1]

    for position, (status, style) in enumerate(STATUS_STYLE.items()):
        selected = visible["preparation_status"].eq(status)

        mask_ax.scatter(
            visible.loc[selected, "timestamp"],
            np.full(int(selected.sum()), position),
            marker="|",
            s=45,
            color=style["color"],
        )

    mask_ax.set_yticks(
        range(len(STATUS_STYLE)),
        [style["label"] for style in STATUS_STYLE.values()],
    )
    mask_ax.set_ylim(-0.6, 2.6)
    mask_ax.set_ylabel("Preparation\nmask")
    mask_ax.set_xlabel("Recorded time")
    mask_ax.grid(axis="x", alpha=0.2)
    format_time_axis(mask_ax)

    handles = [
        Line2D(
            [0],
            [0],
            color="#B8B8B8",
            label="Original observations",
        )
    ] + [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=style["color"],
            label=style["label"],
        )
        for style in STATUS_STYLE.values()
    ]

    axes[0].legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0, 1.25),
        ncol=2,
        fontsize=8,
    )

    fig.suptitle(
        "Experimental data preparation\n"
        f"{len(segment_ids)} retained segments | "
        f"{int(counts['retained']):,} retained rows | "
        f"{bad_time_count} rows without plottable timestamps",
        fontsize=12,
    )

    fig.savefig(
        plot_dir / "preparation_overview.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    if not segment_ids:
        print("No retained segments: overview and counts saved.")
        return

    # ------------------------------------------------------------
    # 2. Gallery: every retained segment, one gallery per channel
    # ------------------------------------------------------------
    ncols = min(3, len(segment_ids))
    nrows = ceil(len(segment_ids) / ncols)

    for channel in channels:
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5 * ncols, 3.3 * nrows),
            squeeze=False,
            constrained_layout=True,
        )

        flat_axes = axes.ravel()

        for ax, segment_id in zip(flat_axes, segment_ids):
            part = visible.loc[visible["segment_id"].eq(segment_id)]

            plot_without_bridging(
                ax,
                part,
                f"{channel}_value",
                max_gap_seconds,
                color="#238B57",
            )

            ax.set_title(segment_title(segment_id, part), fontsize=9)
            ax.set_ylabel(LABELS[channel])
            ax.grid(alpha=0.2)
            format_time_axis(ax)

        for ax in flat_axes[len(segment_ids) :]:
            ax.set_visible(False)

        fig.suptitle(
            f"Retained experimental segments — {LABELS[channel]}\n"
            "Original readings; each panel uses its own axis limits",
            fontsize=12,
        )

        fig.savefig(
            plot_dir / f"retained_gallery_{channel}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

    # ------------------------------------------------------------
    # 3. Individual figures for reporting and closer inspection
    # ------------------------------------------------------------
    for segment_id in segment_ids:
        part = visible.loc[visible["segment_id"].eq(segment_id)]

        fig, axes = plt.subplots(
            len(channels),
            1,
            figsize=(12, max(3.5, 2.8 * len(channels))),
            sharex=True,
            squeeze=False,
            constrained_layout=True,
        )

        axes = axes[:, 0]

        for ax, channel in zip(axes, channels):
            plot_without_bridging(
                ax,
                part,
                f"{channel}_value",
                max_gap_seconds,
                color="#238B57",
            )

            ax.set_ylabel(LABELS[channel])
            ax.grid(alpha=0.2)

        intervals = part["timestamp"].diff().dt.total_seconds().dropna()

        axes[-1].set_xlabel("Recorded time")
        format_time_axis(axes[-1])

        fig.suptitle(
            segment_title(segment_id, part)
            + "\n"
            + (
                f"Sampling intervals: " f"{intervals.min():g}–{intervals.max():g} s"
                if len(intervals)
                else "Single recorded timestamp"
            ),
            fontsize=11,
        )

        fig.savefig(
            detail_dir / f"{segment_id}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

    print(f"Preparation figures saved to: {plot_dir.resolve()}")
