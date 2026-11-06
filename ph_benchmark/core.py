import numpy as np
from scipy.special import logsumexp

from smoothing.filters import (
    EMAFilter,
    WeightedCausalSavitzkyGolayFilter,
)

SCENARIOS = (
    "stable",
    "decline",
    "abrupt_flattening",
    "gradual_flattening",
    "temporary_disturbance",
)


def smooth(values, family, size=1):
    """
    Apply an existing causal filter.

    Weighted SG settings:
        polynomial degree = 1
        oldest sample weight = 0.2
        newest sample weight = 1.0
    """
    values = np.asarray(values, dtype=float)

    if family == "raw":
        return values.copy()

    if family == "ema":
        return EMAFilter(span=size).apply_array(values)

    if family == "wsg":
        return WeightedCausalSavitzkyGolayFilter(
            window_length=size,
            polyorder=1,
            oldest_weight_ratio=0.2,
        ).apply_array(values)

    raise ValueError(f"Unknown smoothing family: {family}")


def block_slopes(values, block=30):
    """
    Calculate slopes from non-overlapping blocks.

    Assumes 1-second sampling.

    A slope becomes available only at the final sample of its block:
        first block: samples 0..29, available at second 29
        second block: samples 30..59, available at second 59

    Returns:
        availability_times: seconds
        slopes: pH units per minute
    """
    values = np.asarray(values, dtype=float)
    n_blocks = len(values) // block

    centered_time = np.arange(block) - (block - 1) / 2

    blocks = values[: n_blocks * block].reshape(
        n_blocks,
        block,
    )

    slopes = blocks @ centered_time / (centered_time @ centered_time) * 60.0

    availability_times = np.arange(n_blocks) * block + block - 1

    return availability_times, slopes


def bocpd_scores(
    values,
    sigma,
    hazard=1 / 20,
    prior_mean=-0.5,
    prior_sd=1.0,
):
    """
    Gaussian mean-change Bayesian online change-point detection.

    Input:
        Normalized block slopes, not raw pH.

    Observation model:
        Gaussian with fixed observation SD.
        Unknown mean with a normal prior.

    Convention:
        A change occurs BEFORE the current observation.
        Run length counts observations including the current one.
        Run lengths therefore range from 1 to t.

    Score:
        Posterior probability that run length is 1 or 2.

    Important:
        - Hazard is per feature block, not per raw sample.
        - This is an exact, unpruned implementation.
        - Gaussian/independence assumptions are provisional.
        - Startup scores are high and must be ignored during warmup.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    if not 0 < hazard < 1:
        raise ValueError("hazard must lie between 0 and 1.")

    prior_variance = prior_sd**2
    observation_variance = sigma**2

    log_probabilities = np.array([])
    means = np.array([])
    variances = np.array([])

    scores = []

    for value in values:

        def log_normal_density(mean, variance):
            return -0.5 * (
                np.log(2 * np.pi * variance) + (value - mean) ** 2 / variance
            )

        # Predictive distribution for a newly started run.
        reset_predictive = log_normal_density(
            prior_mean,
            prior_variance + observation_variance,
        )

        if len(log_probabilities):
            reset_joint = np.log(hazard) + reset_predictive

            growth_joint = (
                log_probabilities
                + np.log1p(-hazard)
                + log_normal_density(
                    means,
                    variances + observation_variance,
                )
            )

            joint = np.r_[reset_joint, growth_joint]

        else:
            joint = np.array([reset_predictive])

        log_probabilities = joint - logsumexp(joint)

        # Update the mean posterior for every possible run.
        old_means = np.r_[prior_mean, means]
        old_variances = np.r_[prior_variance, variances]

        variances = 1.0 / (1.0 / old_variances + 1.0 / observation_variance)

        means = variances * (old_means / old_variances + value / observation_variance)

        recent_change_probability = np.exp(logsumexp(log_probabilities[:2]))

        scores.append(float(recent_change_probability))

    return np.asarray(scores)


def slope_scores(values):
    """
    Baseline change score.

    Compare median slopes in two adjacent three-block histories.

    With 30-second blocks, this compares two adjacent
    90-second periods.
    """
    values = np.asarray(values, dtype=float)
    scores = np.zeros(len(values))

    for i in range(5, len(values)):
        earlier = np.median(values[i - 5 : i - 2])
        recent = np.median(values[i - 2 : i + 1])

        scores[i] = abs(recent - earlier)

    return scores


def alarms(
    times,
    scores,
    threshold,
    warmup=180,
    refractory=120,
):
    """
    Convert scores into rising-edge alarms.

    Alarm timestamps are actual availability times.
    They are not backdated to an estimated change time.
    """
    events = []

    previously_above = False
    last_alarm = -np.inf

    for time, score in zip(times, scores):
        above = score >= threshold

        if (
            time >= warmup
            and above
            and not previously_above
            and time - last_alarm >= refractory
        ):
            events.append(float(time))
            last_alarm = time

        previously_above = above

    return events


def event_metrics(
    events,
    target,
    n=1200,
    warmup=180,
    tolerance=180,
):
    """
    Evaluate alarms against at most one target interval.

    target:
        None for no target event.
        (onset, end) for a target transition.

    Abrupt transition:
        onset == end

    Matching:
        First alarm from onset through end + tolerance.

    Early, late and duplicate alarms count as false alarms.

    Missed events retain NaN delay, never zero delay.
    """
    events = list(events)

    if target is None:
        matched = []
    else:
        matched = [
            event for event in events if target[0] <= event <= target[1] + tolerance
        ]

    detected = bool(matched)

    matched_event = min(matched) if detected else np.nan

    if target is None:
        excluded_exposure = 0
    else:
        excluded_exposure = max(
            0,
            min(n, target[1] + tolerance + 1) - max(warmup, target[0]),
        )

    negative_exposure_hours = (n - warmup - excluded_exposure) / 3600.0

    return {
        "target_count": int(target is not None),
        "detected": int(detected),
        "missed": int(target is not None and not detected),
        "delay_from_onset_seconds": (matched_event - target[0] if detected else np.nan),
        "delay_after_interval_seconds": (
            max(0, matched_event - target[1]) if detected else np.nan
        ),
        "false_alarms": len(events) - int(detected),
        "negative_exposure_hours": negative_exposure_hours,
    }


def generate(
    profile,
    scenario,
    seed,
    noise_multiplier,
    n=1200,
):
    """
    Generate a phenomenological pH scenario.

    This is NOT a chemically validated full-cycle model.

    Data-informed quantities:
        initial pH
        declining slope scale
        residual SD scale
        residual lag-1 correlation
        observed recording grid

    Assumed quantities:
        duration
        transition timing
        transition shape
        disturbance magnitude
        latent AR(1) fluctuation model

    Residual SD and correlation are descriptive measurements,
    not identified sensor-noise parameters.
    """
    rng = np.random.default_rng(seed)

    times = np.arange(n)

    slope = -profile["decline_scale_ph_per_minute"] * rng.uniform(0.75, 1.25) / 60.0

    change_time = int(rng.integers(450, 651))

    initial_level = profile["start_ph"] + rng.uniform(-0.04, 0.04)

    target = None

    if scenario == "stable":
        clean = np.full(n, initial_level)

    elif scenario == "decline":
        clean = initial_level + slope * times

    elif scenario == "abrupt_flattening":
        clean = initial_level + slope * np.minimum(times, change_time)

        target = (change_time, change_time)

    elif scenario == "gradual_flattening":
        transition_duration = int(rng.integers(90, 181))

        transition_time = np.clip(
            times - change_time,
            0,
            transition_duration,
        )

        clean = initial_level + slope * (
            np.minimum(times, change_time)
            + transition_time
            - transition_time**2 / (2 * transition_duration)
        )

        target = (
            change_time,
            change_time + transition_duration,
        )

    elif scenario == "temporary_disturbance":
        disturbance_duration = 90
        normalized_time = (times - change_time) / disturbance_duration

        bump = np.where(
            (normalized_time >= 0) & (normalized_time <= 1),
            np.sin(np.pi * np.clip(normalized_time, 0, 1)) ** 2,
            0,
        )

        clean = initial_level + slope * times + 0.04 * bump

        # This is a nuisance disturbance, not a target endpoint.
        target = None

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    rho = profile["residual_lag1_median"]

    sigma = profile["positive_residual_sd_median"] * noise_multiplier

    fluctuations = np.empty(n)
    fluctuations[0] = rng.normal(0, sigma)

    innovations = rng.normal(
        0,
        sigma * np.sqrt(1 - rho**2),
        n,
    )

    for i in range(1, n):
        fluctuations[i] = rho * fluctuations[i - 1] + innovations[i]

    # Apply rounding once, after adding fluctuations.
    grid = profile["observed_grid_ph"]

    observed = np.round((clean + fluctuations) / grid) * grid

    return times, clean, observed, target
