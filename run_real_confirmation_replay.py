import argparse
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
    read_segment,
    characterize,
    calibrate,
    filtered,
    features,
    bocpd,
    decide,
)
from run_confirmation import confirm


def main():
    parser = argparse.ArgumentParser(
        description="Replay locked K2 and K4 confirmation rules on real S005."
    )
    parser.add_argument("--segment", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    # 1. Verify the input and the settings selected by the earlier experiment.
    lock = json.loads(args.lock.read_text(encoding="utf-8"))

    if lock["filter"] != "wsg" or lock["detector"] != "multi":
        raise ValueError("Expected locked WSG + multivariate BOCPD settings.")

    if lock["baseline"] != "K2_keep" or lock["selected"] != "K4_keep":
        raise ValueError(
            f"Expected K2_keep versus K4_keep. This lock contains "
            f"{lock['baseline']} versus {lock['selected']}."
        )

    digest = hashlib.sha256(args.segment.read_bytes()).hexdigest()
    if digest != lock["source_sha256"]:
        raise ValueError(
            "The segment does not match the CSV used for run_confirmation.py. "
            "Use the original S005 input corresponding to this lock."
        )

    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Choose a new or empty output folder.")

    timestamps, observations = read_segment(args.segment)

    # The earlier lock stored feature scaling and detector settings, but
    # not the pH decline magnitude used by the confirmation rule.
    # Recover that SAME calibration from the verified original source.
    # This is retrospective replay of the calibration run, not external validation.
    profile = calibrate(observations, characterize(observations))
    decline = float(profile["decline"])
    flat_limit = 0.30 * decline

    # 2. Compute the shared signal, features, scores and alarms only once.
    smoothed = filtered(observations, "wsg", int(lock["size"]))
    times, slopes = features(smoothed)

    center = np.asarray(lock["center"], dtype=float)
    scale = np.asarray(lock["scale"], dtype=float)
    threshold = float(lock["threshold"])

    scores = bocpd((slopes - center) / scale)

    raw_alarms, original_k2 = decide(
        times,
        scores,
        threshold,
        slopes[:, 0],  # pH slope is used for confirmation
        "absorption",
        decline,
    )

    # 3. Feed identical alarm times and pH slopes to the two locked rules.
    switches = {
        "K2_keep": confirm(times, slopes[:, 0], raw_alarms, decline, (2, False)),
        "K4_keep": confirm(times, slopes[:, 0], raw_alarms, decline, (4, False)),
    }

    # Check that our comparator reproduces the original pilot decision.
    expected_k2 = original_k2[0] if original_k2 else None
    if switches["K2_keep"] != expected_k2:
        raise AssertionError("K2 replay differs from the original pilot rule.")

    args.out.mkdir(parents=True, exist_ok=True)

    # 4. Save one row per completed 30-second block.
    near_flat = np.abs(slopes[:, 0]) <= flat_limit
    streak = 0
    streaks = []
    for is_flat in near_flat:
        streak = streak + 1 if is_flat else 0
        streaks.append(streak)

    blocks = pd.DataFrame(
        {
            "available_at": timestamps.iloc[times].to_numpy(),
            "elapsed_seconds": times,
            "ph_slope_per_min": slopes[:, 0],
            "temperature_slope_C_per_min": slopes[:, 1],
            "pressure_slope_barg_per_min": slopes[:, 2],
            "bocpd_score": scores,
            "raw_change_alarm": np.isin(times, raw_alarms),
            "near_flat_ph": near_flat,
            "consecutive_flat_blocks": streaks,
        }
    )

    for rule, event in switches.items():
        blocks[f"{rule}_switch"] = (
            times == event if event is not None else np.zeros(len(times), dtype=bool)
        )

    blocks.to_csv(args.out / "block_replay.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "rule": rule,
                "switch_recommended": event is not None,
                "elapsed_seconds": event,
                "recorded_switch_time": (
                    timestamps.iloc[int(event)] if event is not None else pd.NaT
                ),
                "interpretation": (
                    "Unvalidated switching candidate"
                    if event is not None
                    else "No recommendation before segment end"
                ),
            }
            for rule, event in switches.items()
        ]
    )
    summary.to_csv(args.out / "switch_summary.csv", index=False)

    # 5. Plot the measured pH, the block evidence and the detector score.
    minutes = (timestamps - timestamps.iloc[0]).dt.total_seconds().to_numpy() / 60
    block_minutes = times / 60

    fig, axes = plt.subplots(
        3, 1, figsize=(11, 8), sharex=True, constrained_layout=True
    )

    axes[0].plot(
        minutes,
        observations[:, 0],
        color="grey",
        lw=1,
        label="Recorded pH",
    )
    axes[0].plot(
        minutes,
        smoothed[:, 0],
        color="#0072b2",
        lw=1.5,
        label="Causal WSG",
    )
    axes[0].set_ylabel("pH")

    axes[1].axhspan(
        -flat_limit,
        flat_limit,
        color="#009e73",
        alpha=0.15,
        label="Near-flat band",
    )
    axes[1].plot(
        block_minutes,
        slopes[:, 0],
        "o-",
        color="#0072b2",
        markersize=4,
        label="Completed 30-second slope",
    )
    axes[1].axhline(0, color="grey", lw=0.7)
    axes[1].set_ylabel("pH slope\n(pH/min)")

    axes[2].plot(
        block_minutes,
        scores,
        "o-",
        color="#555555",
        markersize=4,
        label="BOCPD score",
    )
    axes[2].axhline(
        threshold,
        color="grey",
        ls="--",
        label="Locked alarm threshold",
    )
    alarm_mask = np.isin(times, raw_alarms)
    axes[2].scatter(
        block_minutes[alarm_mask],
        scores[alarm_mask],
        color="purple",
        s=55,
        zorder=4,
        label="Issued change alarm",
    )
    axes[2].set_ylabel("Recent-change\nprobability")
    axes[2].set_xlabel(f"Minutes since {timestamps.iloc[0]:%H:%M:%S}")

    for rule, color in [("K2_keep", "#d55e00"), ("K4_keep", "#009e73")]:
        event = switches[rule]
        if event is not None:
            for ax in axes:
                ax.axvline(
                    event / 60,
                    color=color,
                    ls=":",
                    lw=1.6,
                    label=f"{rule} recommendation" if ax is axes[0] else None,
                )

    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="best")

    def display_time(event):
        return (
            timestamps.iloc[int(event)].strftime("%H:%M:%S")
            if event is not None
            else "no recommendation"
        )

    fig.suptitle(
        "Real S005 replay: same BOCPD alarms, different confirmation rules\n"
        f"K2: {display_time(switches['K2_keep'])}    |    "
        f"K4: {display_time(switches['K4_keep'])}"
    )
    fig.savefig(args.out / "real_S005_K2_vs_K4.png", dpi=200)
    plt.close(fig)

    print(
        "Raw BOCPD alarms:",
        [timestamps.iloc[int(t)].strftime("%H:%M:%S") for t in raw_alarms],
    )
    print(f"Near-flat band: ±{flat_limit:.5f} pH/min")
    print(f"Completed blocks: {len(times)}")
    print(f"Unused tail for block features: {len(observations) % BLOCK} readings")
    print(summary.to_string(index=False))
    print(f"Saved to: {args.out.resolve()}")


if __name__ == "__main__":
    main()
