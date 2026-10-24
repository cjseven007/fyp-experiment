from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

from .config import UNITS

# Panel order: top to bottom.
PLOT_ORDER = ["co2", "pressure", "temperature", "ph"]

DISPLAY_NAMES = {
    "co2": "CO₂",
    "pressure": "Pressure",
    "temperature": "Temperature",
    "ph": "pH",
}

MODE_COLORS = {
    "Manual": "#D9D9D9",
    "Absorption": "#B8DCC6",
    "Desorption": "#F5C6A5",
    "Unknown": "#D8D1EA",
}

EVENT_COLORS = {
    "startup_zero": "crimson",
    "invalid_reading": "crimson",
    "shared_freeze_suspected": "darkorange",
    "co2_at_limit": "purple",
}

EVENT_NAMES = {
    "startup_zero": "All-channel zero readings",
    "invalid_reading": "Out-of-range or missing reading",
    "shared_freeze_suspected": "Suspected shared-channel freeze",
    "co2_at_limit": "CO₂ at upper measurement limit",
}


def normalize_mode(value):
    """Standardize mode labels without changing the source data."""
    if pd.isna(value):
        return "Unknown"

    text = str(value).strip()

    known = {
        "manual": "Manual",
        "absorption": "Absorption",
        "desorption": "Desorption",
    }

    return known.get(text.lower(), text or "Unknown")


def get_mode_segments(df, max_gap_seconds=3.0):
    """Find contiguous logged operating-mode periods."""
    if "Mode" in df.columns:
        modes = df["Mode"].map(normalize_mode)
    else:
        modes = pd.Series("Unknown", index=df.index)

    dt = df["timestamp"].diff().dt.total_seconds()

    breaks = modes.ne(modes.shift()) | dt.lt(0) | dt.gt(max_gap_seconds)

    groups = breaks.cumsum()
    segments = []

    for _, part in df.groupby(groups, sort=False):
        first = part.index[0]
        last = part.index[-1]

        start = df.loc[first, "timestamp"]
        end = df.loc[last, "timestamp"]

        # Extend shading to the next sample only across a short,
        # forward-time interval. Do not shade across long gaps.
        next_position = last + 1

        if next_position < len(df):
            next_time = df.loc[next_position, "timestamp"]
            gap = (next_time - end).total_seconds()

            if 0 <= gap <= max_gap_seconds:
                end = next_time

        segments.append(
            {
                "start": start,
                "end": end,
                "mode": modes.loc[first],
            }
        )

    return segments


def extract_events(df, max_gap_seconds=3.0):
    """
    Group consecutive flagged rows into individual events.

    Each flag type is grouped separately. Therefore clipping and
    a shared freeze can overlap without becoming one event.
    """
    zero = df["flag_all_zero"].fillna(False).astype(bool)

    invalid_columns = [f"flag_{name}_invalid" for name in PLOT_ORDER]

    invalid = df[invalid_columns].any(axis=1) & ~zero

    masks = {
        "startup_zero": zero,
        "invalid_reading": invalid,
        "shared_freeze_suspected": (
            df["flag_shared_freeze_suspected"].fillna(False).astype(bool)
        ),
        "co2_at_limit": (df["flag_co2_at_limit"].fillna(False).astype(bool)),
    }

    dt = df["timestamp"].diff().dt.total_seconds()
    time_break = dt.lt(0) | dt.gt(max_gap_seconds)

    events = []

    for event_type, mask in masks.items():
        starts = mask.ne(mask.shift(fill_value=False)) | time_break

        groups = starts.cumsum()

        for _, part in df.loc[mask].groupby(groups.loc[mask], sort=False):
            start = part["timestamp"].iloc[0]
            end = part["timestamp"].iloc[-1]

            if event_type == "invalid_reading":
                channels = [
                    DISPLAY_NAMES[name]
                    for name in PLOT_ORDER
                    if part[f"flag_{name}_invalid"].any()
                ]
            elif event_type == "co2_at_limit":
                channels = ["CO₂"]
            elif event_type == "shared_freeze_suspected":
                channels = ["Pressure", "Temperature", "pH"]
            else:
                channels = [DISPLAY_NAMES[name] for name in PLOT_ORDER]

            events.append(
                {
                    "event_type": event_type,
                    "start": start,
                    "end": end,
                    "duration_seconds": max(0.0, (end - start).total_seconds()),
                    "rows_flagged": len(part),
                    "channels": ", ".join(channels),
                    "source_row_start": int(part["source_row"].iloc[0]),
                    "source_row_end": int(part["source_row"].iloc[-1]),
                }
            )

    events.sort(key=lambda event: (event["start"], event["event_type"]))

    for number, event in enumerate(events, start=1):
        event["event_id"] = f"E{number:03d}"

    return events


def draw_panel_figure(
    df,
    output_path,
    title,
    mode_segments,
    focus_event=None,
    view_start=None,
    view_end=None,
):
    """Draw one overview or event-detail figure."""
    if view_start is None:
        view_start = df["timestamp"].min()

    if view_end is None:
        view_end = df["timestamp"].max()

    # Avoid an identical left/right axis limit for a single timestamp.
    if view_end <= view_start:
        view_end = view_start + pd.Timedelta(seconds=1)

    visible = df.loc[df["timestamp"].between(view_start, view_end)].copy()

    if visible.empty:
        return False

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(14, 10),
        sharex=True,
        constrained_layout=True,
    )

    modes_shown = []

    # Shade modes consistently across every panel.
    for segment in mode_segments:
        left = max(segment["start"], view_start)
        right = min(segment["end"], view_end)

        if right <= left:
            continue

        mode = segment["mode"]
        color = MODE_COLORS.get(mode, MODE_COLORS["Unknown"])

        if mode not in modes_shown:
            modes_shown.append(mode)

        for ax in axes:
            ax.axvspan(
                left,
                right,
                color=color,
                alpha=0.28,
                linewidth=0,
                zorder=0,
            )

        # Label periods wide enough to fit text.
        fraction = (right - left) / (view_end - view_start)

        if fraction >= 0.08:
            midpoint = left + (right - left) / 2

            axes[0].text(
                midpoint,
                0.98,
                mode,
                transform=axes[0].get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=9,
                bbox={
                    "facecolor": "white",
                    "alpha": 0.8,
                    "edgecolor": "none",
                    "pad": 2,
                },
            )

    for ax, name in zip(axes, PLOT_ORDER):
        values = visible[f"{name}_value"]

        # Break the connecting line at timestamp gaps/reversals.
        # Scatter points preserve the actual observed readings.
        line_values = values.copy()
        dt = visible["timestamp"].diff().dt.total_seconds()
        line_values.loc[dt.gt(3.0) | dt.lt(0)] = float("nan")

        ax.plot(
            visible["timestamp"],
            line_values,
            color="steelblue",
            linewidth=0.8,
            label="Recorded values",
            zorder=2,
        )

        ax.scatter(
            visible["timestamp"],
            values,
            color="steelblue",
            s=3,
            zorder=2,
        )

        invalid = visible[f"flag_{name}_invalid"] | visible["flag_all_zero"]

        ax.scatter(
            visible.loc[invalid, "timestamp"],
            values.loc[invalid],
            color="crimson",
            s=18,
            marker="x",
            label="Invalid / all-zero flag",
            zorder=5,
        )

        # Missing values have no y coordinate; mark their time at
        # the bottom of the relevant panel instead.
        missing = values.isna()

        if missing.any():
            ax.plot(
                visible.loc[missing, "timestamp"],
                [0.03] * int(missing.sum()),
                linestyle="none",
                marker="x",
                color="crimson",
                transform=ax.get_xaxis_transform(),
                label="Missing numeric reading",
                zorder=5,
            )

        if name in ("pressure", "temperature", "ph"):
            frozen = visible["flag_shared_freeze_suspected"]

            ax.scatter(
                visible.loc[frozen, "timestamp"],
                values.loc[frozen],
                color="darkorange",
                s=10,
                label="Suspected shared freeze",
                zorder=4,
            )

        if name == "co2":
            clipped = visible["flag_co2_at_limit"]

            ax.scatter(
                visible.loc[clipped, "timestamp"],
                values.loc[clipped],
                color="purple",
                s=10,
                label="At sensor upper limit",
                zorder=4,
            )

        if focus_event is not None:
            color = EVENT_COLORS[focus_event["event_type"]]

            # Mark event boundaries without suggesting a chemical
            # change point has been confirmed.
            for boundary in (focus_event["start"], focus_event["end"]):
                if view_start <= boundary <= view_end:
                    ax.axvline(
                        boundary,
                        color=color,
                        linestyle="--",
                        linewidth=1.1,
                        alpha=0.9,
                    )

        ax.set_ylabel(f"{DISPLAY_NAMES[name]}\n({UNITS[name]})")
        ax.grid(alpha=0.2)
        ax.legend(loc="lower left", fontsize=7)
        ax.margins(y=0.18)

    mode_handles = [
        Patch(
            facecolor=MODE_COLORS.get(mode, MODE_COLORS["Unknown"]),
            alpha=0.5,
            label=mode,
        )
        for mode in modes_shown
    ]

    if mode_handles:
        fig.legend(
            handles=mode_handles,
            loc="outside lower center",
            ncol=min(4, len(mode_handles)),
            title="Logged operating mode",
            fontsize=9,
        )

    axes[-1].set_xlim(view_start, view_end)
    axes[-1].set_xlabel("Recorded time")

    fig.suptitle(title, fontsize=12)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return True


def create_audit_plots(
    df,
    output_dir,
    context_seconds=60,
    boundary_seconds=60,
):
    """
    Save:
      - overview
      - one contextual figure per event
      - start/end details for long events
      - event index CSV

    context_seconds:
        Additional time before/after an event.

    boundary_seconds:
        Half-width of close-ups around long-event boundaries.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zoom_dir = output_dir / "event_details"
    zoom_dir.mkdir(exist_ok=True)

    # Keep source order. Reset only the pandas index.
    visible = df.loc[df["timestamp"].notna()].copy().reset_index(drop=True)

    if visible.empty:
        raise ValueError("No valid timestamps available for plotting.")

    mode_segments = get_mode_segments(visible)
    events = extract_events(visible)

    draw_panel_figure(
        visible,
        output_dir / "audit_overview.png",
        title=(
            "Experimental data audit\n" "Mode shading and sensor-quality annotations"
        ),
        mode_segments=mode_segments,
    )

    records = []

    for event in events:
        event_id = event["event_id"]
        event_type = event["event_type"]
        label = EVENT_NAMES[event_type]

        safe_type = re.sub(r"[^a-zA-Z0-9_]", "_", event_type)
        prefix = f"{event_id}_{safe_type}"

        title = (
            f"{event_id}: {label}\n"
            f"{event['start']:%Y-%m-%d %H:%M:%S} to "
            f"{event['end']:%H:%M:%S} | "
            f"Elapsed span: {event['duration_seconds']:.0f} s"
        )

        context_file = f"{prefix}_context.png"

        draw_panel_figure(
            visible,
            zoom_dir / context_file,
            title=title,
            mode_segments=mode_segments,
            focus_event=event,
            view_start=event["start"] - pd.Timedelta(seconds=context_seconds),
            view_end=event["end"] + pd.Timedelta(seconds=context_seconds),
        )

        record = {
            **event,
            "context_figure": context_file,
            "start_figure": "",
            "end_figure": "",
        }

        # A long event needs boundary close-ups as well as context.
        if event["duration_seconds"] > 2 * boundary_seconds:
            for boundary_name in ("start", "end"):
                center = event[boundary_name]
                filename = f"{prefix}_{boundary_name}.png"

                draw_panel_figure(
                    visible,
                    zoom_dir / filename,
                    title=(
                        f"{event_id}: {label} — "
                        f"{boundary_name} detail\n"
                        f"Boundary: {center:%Y-%m-%d %H:%M:%S}"
                    ),
                    mode_segments=mode_segments,
                    focus_event=event,
                    view_start=center - pd.Timedelta(seconds=boundary_seconds),
                    view_end=center + pd.Timedelta(seconds=boundary_seconds),
                )

                record[f"{boundary_name}_figure"] = filename

        records.append(record)

    columns = [
        "event_id",
        "event_type",
        "start",
        "end",
        "duration_seconds",
        "rows_flagged",
        "channels",
        "source_row_start",
        "source_row_end",
        "context_figure",
        "start_figure",
        "end_figure",
    ]

    pd.DataFrame(records, columns=columns).to_csv(
        output_dir / "event_index.csv",
        index=False,
    )

    print(f"Generated overview and details for {len(events)} events.")
