import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import lfilter
from scipy.special import logsumexp

# Column order in every signal matrix.
CH = ["ph", "temperature", "pressure"]
UNITS = ["pH", "°C", "barg"]
CASES = ["A_decline_or_rise", "B_flattening", "C_pressure_pulse"]

# Assumed recording grids for this S005-specific pilot.
# These are not certified instrument accuracies.
GRID = np.array([0.01, 0.01, 0.01])

# Times in seconds; the script requires 1 Hz data.
BLOCK = 30
WARMUP = 180
TOL = 180


# ------------------------------------------------------------
# STEP 1: READ AND CHARACTERISE THE REAL SEGMENT
# ------------------------------------------------------------


def read_segment(path):
    d = pd.read_csv(path)
    ts = pd.to_datetime(d["timestamp"], errors="raise")

    if ts.isna().any() or not ts.diff().iloc[1:].eq(pd.Timedelta(seconds=1)).all():
        raise ValueError("Use one contiguous 1 Hz segment. Do not silently fill gaps.")

    for ch in CH:
        ok = d["usable_" + ch].astype(str).str.strip().str.lower()
        if not ok.isin(["true", "1", "1.0"]).all():
            raise ValueError(f"Unusable {ch} readings in segment.")

    y = d[[ch + "_value" for ch in CH]].to_numpy(float)

    if len(y) < 600 or not np.isfinite(y).all():
        raise ValueError("This S005 pilot needs at least 600 finite rows.")

    return ts, y


def characterize(y):
    """Same complete 60-second windows for all three channels."""
    rows = []
    x = np.arange(60) - 29.5

    for i in range(len(y) // 60):
        block = y[60 * i : 60 * (i + 1)]

        # Three local linear fits, one per column.
        slopes = x @ block / (x @ x)
        errors = block - block.mean(axis=0) - x[:, None] * slopes

        for j, ch in enumerate(CH):
            e = errors[:, j].copy()
            v = block[:, j]

            if np.max(np.abs(e)) < 1e-10:
                e[:] = 0

            if min(e[:-1].std(), e[1:].std()) > 1e-10:
                rho = np.corrcoef(e[:-1], e[1:])[0, 1]
            else:
                rho = np.nan

            rows.append(
                [
                    i + 1,
                    ch,
                    slopes[j] * 60,
                    e.std(ddof=1),
                    rho,
                    np.mean(np.diff(v) == 0),
                ]
            )

    return pd.DataFrame(
        rows,
        columns=[
            "window",
            "channel",
            "slope_per_min",
            "residual_sd",
            "lag1",
            "unchanged",
        ],
    )


def calibrate(y, stats):
    """Extract descriptive S005 values used by the generator."""
    med = stats.groupby("channel").median().loc[CH]

    ph = stats[stats.channel.eq("ph")]
    decline = -ph.loc[ph.slope_per_min < -1e-8, "slope_per_min"].median()

    if not np.isfinite(decline) or decline <= 0:
        raise ValueError("No measurable declining pH windows.")

    return {
        "start": y[0],
        "decline": decline,  # positive magnitude, pH/min
        "heat": med.loc["temperature", "slope_per_min"],
        "pressure": np.median(y[:, 2]),
        "span": np.ptp(y, axis=0),
        "sd": med.residual_sd.to_numpy(),
        "rho": med.lag1.fillna(0).clip(-0.95, 0.95).to_numpy(),
    }


# ------------------------------------------------------------
# STEP 2: GENERATE PAIRED MULTIVARIATE CASES
# ------------------------------------------------------------


def simulate(p, seed, phase):
    """
    Yield three paired cases for one seed and phase.

    A and B: identical temperature and pressure.
    A and C: identical pH and temperature.

    Returns:
        case name, clean signal, observed signal,
        target time or None, pressure pulse start or None.
    """
    rng = np.random.default_rng([seed, 0 if phase == "absorption" else 1])

    t = np.arange(600)  # ten-minute synthetic segment
    cp = int(rng.integers(245, 351))
    pulse_start = int(rng.integers(245, 351))

    sign = -1 if phase == "absorption" else 1
    slope = sign * p["decline"] * rng.uniform(0.75, 1.25) / 60
    start_ph = p["start"][0] + rng.uniform(-0.03, 0.03)

    # Temperature: a gradual ramp plus a smooth fluctuation.
    # Scale informed by S005; exact shape and periods are assumptions.
    temperature = (
        p["start"][1]
        + p["heat"] * rng.uniform(0.6, 1.15) * t / 60
        + 0.05 * p["span"][1] * np.sin(2 * np.pi * t / rng.uniform(100, 220))
    )

    # Pressure: variation around an observed operating level.
    pressure = p["pressure"] + 0.12 * p["span"][2] * np.sin(
        2 * np.pi * t / rng.uniform(70, 170) + rng.uniform(0, 2 * np.pi)
    )

    # Temporary pressure increase and recovery over 90 seconds.
    u = np.clip((t - pulse_start) / 90, 0, 1)
    bump = 0.5 * p["span"][2] * np.sin(np.pi * u) ** 2

    # Correlated fluctuations. A multiplier of zero isolates rounding.
    # Residual SD/rho are sensitivity inputs, not proven sensor-noise parameters.
    noise = np.zeros((len(t), 3))
    strength = rng.choice([0.0, 0.5, 1.0])

    for j, (sd, rho) in enumerate(zip(p["sd"], p["rho"])):
        eps = rng.normal(size=len(t)) * strength * sd * np.sqrt(1 - rho * rho)
        eps[0] = rng.normal() * strength * sd
        noise[:, j] = lfilter([1], [1, -rho], eps)

    for case in CASES:
        flat = case == CASES[1]
        disturbed = case == CASES[2]

        clean = np.column_stack(
            [
                start_ph + slope * (np.minimum(t, cp) if flat else t),
                temperature,
                pressure + (bump if disturbed else 0),
            ]
        )

        observed = np.round((clean + noise) / GRID) * GRID

        yield (
            case,
            clean,
            observed,
            cp if flat else None,
            pulse_start if disturbed else None,
        )


def sample_set(profile, seeds, phases):
    for seed in seeds:
        for phase in phases:
            for case, clean, obs, cp, pulse in simulate(profile, seed, phase):
                yield seed, phase, case, clean, obs, cp, pulse


# ------------------------------------------------------------
# STEP 4A: CAUSAL SMOOTHING AND SLOPE FEATURES
# ------------------------------------------------------------


def filtered(y, family, size):
    """All filters use only present and past readings."""
    if family == "raw":
        return y.copy()

    if family == "ema":
        return pd.DataFrame(y).ewm(span=size, adjust=False).mean().to_numpy()

    # One-sided weighted SG: fit a degree-1 polynomial and
    # evaluate it at the newest observation, x=0.
    x = np.arange(1 - size, 1, dtype=float)
    X = np.column_stack([np.ones(size), x])
    w = np.geomspace(0.2, 1, size)

    h = np.linalg.solve(
        X.T @ (w[:, None] * X),
        X.T * w,
    )[0]

    out = y.copy()
    for j in range(y.shape[1]):
        out[size - 1 :, j] = np.convolve(y[:, j], h[::-1], mode="valid")
    return out


def features(y):
    """Return one three-channel slope vector per completed block."""
    n = len(y) // BLOCK
    x = np.arange(BLOCK) - (BLOCK - 1) / 2
    blocks = y[: n * BLOCK].reshape(n, BLOCK, -1)

    slopes = np.einsum("i,kij->kj", x, blocks) * 60 / (x @ x)

    # A block containing seconds 0..29 is available at second 29.
    times = np.arange(1, n + 1) * BLOCK - 1
    return times, slopes


# ------------------------------------------------------------
# STEP 4B: BASELINE, UNIVARIATE BOCPD AND MULTIVARIATE BOCPD
# ------------------------------------------------------------


def bocpd(z):
    """
    Gaussian mean-change BOCPD on standardised slope features.

    One shared run length; conditionally independent channels.
    Normal mean prior N(0,I); fixed unit observation variance.
    The hazard 0.05 is per completed block, not per raw second.
    """
    log_prob = np.empty(0)
    mean = np.empty((0, z.shape[1]))
    var = np.empty(0)
    result = []

    for value in z:
        # First row: new run. Other rows: continue existing runs.
        m = np.vstack([np.zeros(z.shape[1]), mean])
        a = np.r_[1.0, var]

        log_predictive = -0.5 * (
            z.shape[1] * np.log(2 * np.pi * (1 + a))
            + ((value - m) ** 2).sum(axis=1) / (1 + a)
        )

        if log_prob.size:
            log_prior = np.r_[np.log(0.05), log_prob + np.log(0.95)]
        else:
            log_prior = np.array([0.0])

        joint = log_prior + log_predictive
        log_prob = joint - logsumexp(joint)

        var = a / (1 + a)
        mean = (m + a[:, None] * value) / (1 + a[:, None])

        # Recent-change score: posterior mass on run lengths 1 and 2.
        result.append(np.exp(logsumexp(log_prob[:2])))

    return np.asarray(result)


def scores(z):
    baseline = np.zeros(len(z))

    for i in range(5, len(z)):
        recent = np.median(z[i - 2 : i + 1, 0])
        previous = np.median(z[i - 5 : i - 2, 0])
        baseline[i] = abs(recent - previous)

    return {
        "slope": baseline,  # pH-only threshold baseline
        "uni": bocpd(z[:, :1]),  # pH-only BOCPD
        "multi": bocpd(z),  # pH + temperature + pressure BOCPD
    }


def decide(times, score, threshold, ph_slopes, phase, decline):
    """
    Return two outputs:
      raw: generic change alarms
      switches: confirmed endpoint-like recommendations

    The decision rule knows the operating mode, not the target time.
    """
    direction = -1 if phase == "absorption" else 1
    raw, switches = [], []
    last_alarm = -999
    pending_until = -1

    for i, t in enumerate(times):
        if t < WARMUP:
            continue

        crossing = score[i] >= threshold and score[i - 1] < threshold

        if crossing and t - last_alarm >= 120:
            raw.append(int(t))
            last_alarm = t

            prior = direction * ph_slopes[max(0, i - 5) : i]
            if np.any(prior > 0.5 * decline):
                pending_until = t + 120

        # A candidate needs two consecutive nearly-flat completed blocks.
        # The switch time is when confirmation becomes available.
        if not switches and t <= pending_until and i >= 1:
            recent = ph_slopes[i - 1 : i + 1]
            if np.all(np.abs(recent) <= 0.30 * decline):
                switches.append(int(t))

    return raw, switches


def evaluate(events, cp):
    matched = [] if cp is None else [t for t in events if cp <= t <= cp + TOL]

    return {
        "target": int(cp is not None),
        "hit": int(bool(matched)),
        "false_alarms": len(events) - int(bool(matched)),
        "delay_s": matched[0] - cp if matched else np.nan,
        "premature": int(cp is not None and any(t < cp for t in events)),
    }


# ------------------------------------------------------------
# RUN THE FOUR STEPS
# ------------------------------------------------------------


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--segment", required=True, type=Path)
    cli.add_argument("--out", required=True, type=Path)
    cli.add_argument("--dev", type=int, default=8)
    cli.add_argument("--test", type=int, default=20)
    cli.add_argument("--both-phases", action="store_true")
    cli.add_argument("--test-start", type=int, default=100000)
    args = cli.parse_args()

    if min(args.dev, args.test) < 1 or args.test_start < 1000 + args.dev:
        cli.error("Use positive repetitions and non-overlapping seed ranges.")

    out = args.out
    if out.exists() and any(out.iterdir()):
        cli.error("Choose a new or empty output directory.")

    ts, real = read_segment(args.segment)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Real-data results.
    stats = characterize(real)
    stats["window_start"] = [ts.iloc[(i - 1) * 60] for i in stats.window]
    stats.to_csv(out / "real_windows.csv", index=False)

    p = calibrate(real, stats)
    phases = ["absorption", "desorption"] if args.both_phases else ["absorption"]
    dev = list(sample_set(p, range(1000, 1000 + args.dev), phases))

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 6), layout="constrained")
    for j, ax in enumerate(axes):
        ax.plot(np.arange(len(real)) / 60, real[:, j])
        ax.set_ylabel(f"{CH[j]} ({UNITS[j]})")
        ax.axvspan(len(real) // 60, (len(real) - 1) / 60, color="grey", alpha=0.15)

    axes[-1].set_xlabel("Minutes; grey = tail excluded from window statistics")
    fig.savefig(out / "real_segment.png", dpi=160)
    plt.close(fig)

    # 2 and 3. Generate one fixed illustrative trio and compare with S005.
    comparison = [stats.assign(case="real_S005")]
    labels = []
    fig, axes = plt.subplots(
        3,
        3,
        sharex=True,
        sharey="row",
        figsize=(13, 7),
        layout="constrained",
    )

    for k, (case, clean, obs, cp, pulse) in enumerate(simulate(p, 42, "absorption")):
        t = np.arange(len(obs))
        frame = pd.DataFrame(obs, columns=[ch + "_value" for ch in CH])
        frame.insert(0, "elapsed_seconds", t)
        frame.to_csv(out / (case + ".csv"), index=False)

        labels.append(
            {
                "case": case,
                "target_s": cp,
                "nuisance_start_s": pulse,
                "nuisance_end_s": None if pulse is None else pulse + 90,
            }
        )

        comparison.append(characterize(obs).assign(case=case))

        for j in range(3):
            ax = axes[j, k]
            ax.plot(t / 60, obs[:, j], lw=0.8, color="steelblue")
            ax.plot(t / 60, clean[:, j], lw=1, color="black", ls="--")
            if cp is not None:
                ax.axvline(cp / 60, color="green")
            if pulse is not None:
                ax.axvspan(pulse / 60, (pulse + 90) / 60, color="orange", alpha=0.18)
            if j == 0:
                ax.set_title(case)
            if k == 0:
                ax.set_ylabel(UNITS[j])
            if j == 2:
                ax.set_xlabel("Minutes")

    fig.suptitle("Blue: observed | dashed: clean | green: target | orange: disturbance")
    fig.savefig(out / "paired_pilot.png", dpi=160)
    plt.close(fig)

    pd.DataFrame(labels).to_csv(out / "pilot_labels.csv", index=False)
    metrics = ["slope_per_min", "residual_sd", "lag1", "unchanged"]
    (
        pd.concat(comparison)
        .groupby(["case", "channel"])[metrics]
        .median()
        .to_csv(out / "pilot_vs_real.csv")
    )

    # 4. Tune smoothers on DEVELOPMENT traces only.
    grid = {"raw": [1], "ema": [5, 15, 31], "wsg": [15, 31, 61]}
    err_scale = np.maximum(p["sd"], GRID / np.sqrt(12))
    tune = []

    for family, sizes in grid.items():
        for size in sizes:
            errors = []
            for _, _, _, clean, obs, _, _ in dev:
                e = (filtered(obs, family, size) - clean)[WARMUP:]
                channel_rmse = np.sqrt(np.mean(e**2, axis=0))
                errors.append(channel_rmse / err_scale)

            tune.append(
                {
                    "family": family,
                    "size": size,
                    "loss": np.mean(errors),
                }
            )

    tune = pd.DataFrame(tune)
    tune.to_csv(out / "filter_tuning.csv", index=False)

    test_rows, rmse_rows, settings, replay = [], [], [], []

    for family, group in tune.groupby("family"):
        size = int(group.loc[group.loss.idxmin(), "size"])
        histories = []

        for _, phase, case, _, obs, cp, _ in dev:
            times, f = features(filtered(obs, family, size))
            histories.append((phase, case, cp, times, f))

        # Fit feature scaling using development Case A only.
        # Never use future values of the real/test trace to normalise it.
        null = np.vstack(
            [
                f[times >= WARMUP]
                for _, case, _, times, f in histories
                if case == CASES[0]
            ]
        )
        center = null.mean(axis=0)
        scale = np.maximum(null.std(axis=0), 1e-6)

        threshold_grid = {
            "slope": [0.5, 1, 1.5, 2],
            "uni": [0.3, 0.5, 0.7, 0.9],
            "multi": [0.3, 0.5, 0.7, 0.9],
        }
        losses = {
            (method, h): []
            for method, candidates in threshold_grid.items()
            for h in candidates
        }

        for phase, _, cp, times, f in histories:
            for method, sc in scores((f - center) / scale).items():
                for h in threshold_grid[method]:
                    _, switches = decide(times, sc, h, f[:, 0], phase, p["decline"])
                    ev = evaluate(switches, cp)
                    loss = (
                        3 * (ev["target"] - ev["hit"])
                        + ev["false_alarms"]
                        + (ev["delay_s"] / 600 if ev["hit"] else 0)
                    )
                    losses[method, h].append(loss)

        best = {
            method: min(candidates, key=lambda h: np.mean(losses[method, h]))
            for method, candidates in threshold_grid.items()
        }

        for method, h in best.items():
            settings.append(
                {
                    "family": family,
                    "size": size,
                    "method": method,
                    "threshold": h,
                    "center": center.tolist(),
                    "scale": scale.tolist(),
                }
            )

        # Frozen settings; new test seeds.
        test_seeds = range(args.test_start, args.test_start + args.test)

        for seed, phase, case, clean, obs, cp, _ in sample_set(p, test_seeds, phases):
            sm = filtered(obs, family, size)
            rmse = np.sqrt(np.mean((sm[WARMUP:] - clean[WARMUP:]) ** 2, axis=0))

            keys = dict(seed=seed, phase=phase, case=case, family=family)
            rmse_rows.append({**keys, **dict(zip(CH, rmse))})

            times, f = features(sm)
            for method, sc in scores((f - center) / scale).items():
                raw, switches = decide(
                    times, sc, best[method], f[:, 0], phase, p["decline"]
                )
                for layer, events in [("change", raw), ("switch", switches)]:
                    test_rows.append(
                        {
                            **keys,
                            "method": method,
                            "layer": layer,
                            "alarms_s": events,
                            **evaluate(events, cp),
                        }
                    )

        # Real S005 replay: report timestamps, never physical endpoint accuracy.
        times, f = features(filtered(real, family, size))
        for method, sc in scores((f - center) / scale).items():
            raw, switches = decide(
                times, sc, best[method], f[:, 0], "absorption", p["decline"]
            )
            for layer, events in [("change", raw), ("switch", switches)]:
                replay.append(
                    {
                        "family": family,
                        "method": method,
                        "layer": layer,
                        "alarms_s": events,
                        "recorded_times": [str(ts.iloc[t]) for t in events],
                    }
                )

    pd.DataFrame(settings).to_csv(out / "frozen_settings.csv", index=False)
    pd.DataFrame(replay).to_csv(out / "real_replay.csv", index=False)
    pd.DataFrame(rmse_rows).to_csv(out / "test_rmse.csv", index=False)

    trials = pd.DataFrame(test_rows)
    trials.to_csv(out / "test_events.csv", index=False)

    summary = (
        trials.groupby(["phase", "case", "family", "method", "layer"])
        .agg(
            n=("seed", "size"),
            targets=("target", "sum"),
            hits=("hit", "sum"),
            false_per_trace=("false_alarms", "mean"),
            premature=("premature", "sum"),
            median_detected_delay_s=("delay_s", "median"),
        )
        .reset_index()
    )

    summary["missed"] = summary.targets - summary.hits
    summary["recall"] = summary.hits / summary.targets.replace(0, np.nan)
    summary["recall_mcse"] = np.sqrt(
        summary.recall * (1 - summary.recall) / summary.targets.replace(0, np.nan)
    )

    summary.to_csv(out / "test_summary.csv", index=False)
    print("Done:", out.resolve())


if __name__ == "__main__":
    main()
