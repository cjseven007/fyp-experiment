"""Causal filters and detectors with synthetic labels kept out of inference."""
import numpy as np
from scipy.special import logsumexp
from smoothing.filters import EMAFilter, WeightedCausalSavitzkyGolayFilter


def smooth(y, family, size=1):
    if family == 'raw':
        return np.asarray(y, float).copy()
    if family == 'ema':
        return EMAFilter(size).apply_array(y)
    if family == 'wsg':
        return WeightedCausalSavitzkyGolayFilter(size, 1, 0.2).apply_array(y)
    raise ValueError(family)


def block_slopes(y, block=30):
    """Non-overlapping 30-s slopes, available at the LAST sample of each block.

    Assumes validated 1-Hz input. Returns pH/min; never uses a future block.
    """
    y = np.asarray(y, float)
    n = len(y) // block
    x = np.arange(block) - (block - 1) / 2
    return np.arange(n) * block + block - 1, y[:n * block].reshape(n, block) @ x / (x @ x) * 60


def bocpd_scores(z, sigma, hazard=1/20, prior_mean=-0.5, prior_sd=1.0):
    """Gaussian mean-change BOCPD, known observation SD, normal mean prior.

    Reset occurs BEFORE the current observation. Run length counts observations
    including the current one (1..t). Reset branch uses the PRIOR predictive;
    growth branches use their run-specific predictives. Exact, unpruned recursion.
    Score = posterior mass on run lengths 1 and 2 (recent change), not p(r=0).
    Fixed hazard is per 30-second feature, NOT per raw sample. Gaussian/conditional
    independence assumptions are provisional, especially after filtering/rounding.
    """
    if sigma <= 0 or not 0 < hazard < 1:
        raise ValueError('Require sigma > 0 and 0 < hazard < 1')
    v0, obsvar = prior_sd ** 2, sigma ** 2
    logp, means, variances = np.array([]), np.array([]), np.array([])
    scores = []
    for value in z:
        def logpdf(m, v):
            return -0.5 * (np.log(2 * np.pi * v) + (value - m) ** 2 / v)
        reset = logpdf(prior_mean, v0 + obsvar)
        if len(logp):
            joint = np.r_[np.log(hazard) + reset,
                          logp + np.log1p(-hazard) + logpdf(means, variances + obsvar)]
        else:
            joint = np.array([reset])
        logp = joint - logsumexp(joint)
        old_m, old_v = np.r_[prior_mean, means], np.r_[v0, variances]
        variances = 1 / (1 / old_v + 1 / obsvar)
        means = variances * (old_m / old_v + value / obsvar)
        scores.append(float(np.exp(logsumexp(logp[:2]))))
    return np.asarray(scores)


def slope_scores(z):
    """Difference between medians of two adjacent three-block histories."""
    out = np.zeros(len(z))
    for i in range(5, len(z)):
        out[i] = abs(np.median(z[i-2:i+1]) - np.median(z[i-5:i-2]))
    return out


def alarms(times, scores, threshold, warmup=180, refractory=120):
    """Rising-edge alarms after warmup; no backdating, no truth input."""
    events, previous, last = [], False, -np.inf
    for t, score in zip(times, scores):
        active = score >= threshold
        if t >= warmup and active and not previous and t - last >= refractory:
            events.append(float(t)); last = t
        previous = active
    return events


def event_metrics(events, target, n=1200, warmup=180, tolerance=180):
    """One target interval at most; early, duplicate and late alarms are false.

    For gradual change, delay is from transition ONSET; also report distance
    beyond its end. False alarm denominator excludes the target matching interval.
    Null delay remains NaN for misses. No-change/disturbance trials have no target.
    """
    events = list(events)
    matched = [] if target is None else [e for e in events if target[0] <= e <= target[1] + tolerance]
    detected = bool(matched)
    matched_event = min(matched) if detected else np.nan
    excluded = 0 if target is None else max(0, min(n, target[1]+tolerance+1)-max(warmup,target[0]))
    exposure = (n-warmup-excluded) / 3600
    return dict(target_count=int(target is not None), detected=int(detected),
                missed=int(target is not None and not detected),
                delay_from_onset_seconds=matched_event-target[0] if detected else np.nan,
                delay_after_interval_seconds=max(0,matched_event-target[1]) if detected else np.nan,
                false_alarms=len(events)-int(detected), negative_exposure_hours=exposure)


SCENARIOS = ('stable', 'decline', 'abrupt_flattening', 'gradual_flattening', 'temporary_disturbance')


def generate(profile, scenario, seed, noise_multiplier, n=1200):
    """Phenomenological signals, NOT a chemical full-cycle simulator.

    Observed residual SD/rho inform sensitivity settings; they are not identified
    latent sensor-noise parameters. Quantization is applied once after adding AR(1)
    variation. Zero multiplier isolates rounding alone. All time/shape assumptions
    are documented and varied independently of the generating noise realization.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    slope = -profile['decline_scale_ph_per_minute'] * rng.uniform(0.75, 1.25) / 60
    cp = int(rng.integers(450, 651))
    level = profile['start_ph'] + rng.uniform(-0.04, 0.04)
    target = None
    if scenario == 'stable':
        clean = np.full(n, level)
    elif scenario == 'decline':
        clean = level + slope*t
    elif scenario == 'abrupt_flattening':
        clean = level + slope*np.minimum(t,cp)
        target = (cp,cp)
    elif scenario == 'gradual_flattening':
        width = int(rng.integers(90,181))
        u = np.clip(t-cp,0,width)
        clean = level + slope*(np.minimum(t,cp) + u-u*u/(2*width))
        target = (cp,cp+width)
    elif scenario == 'temporary_disturbance':
        # Nuisance excursion; not a desired endpoint event. Both edges may alarm.
        width = 90
        u = (t-cp)/width
        bump = np.where((u>=0)&(u<=1), np.sin(np.pi*np.clip(u,0,1))**2, 0)
        clean = level+slope*t+0.04*bump
    else:
        raise ValueError(scenario)
    rho = profile['residual_lag1_median']
    sigma = profile['positive_residual_sd_median'] * noise_multiplier
    e = np.empty(n); e[0] = rng.normal(0,sigma)
    innovations = rng.normal(0,sigma*np.sqrt(1-rho*rho),n)
    for i in range(1,n):
        e[i] = rho*e[i-1]+innovations[i]
    step = profile['observed_grid_ph']
    observed = np.round((clean+e)/step)*step
    return t, clean, observed, target
