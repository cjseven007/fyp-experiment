from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NoiseProfile:
    """Estimated measurement-noise characteristics for one real signal."""

    sigma: float
    rho: float
    spike_probability: float
    spike_scale: float
    quantization_step: float | None


def robust_sigma(values: np.ndarray) -> float:
    """Estimate standard deviation using Median Absolute Deviation (MAD).

    MAD is more robust than ordinary standard deviation when occasional spikes
    are present.
    """

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0

    median = np.median(values)
    mad = np.median(np.abs(values - median))
    sigma = 1.4826 * mad

    # Fallback for highly quantized signals where MAD may become exactly zero.
    if sigma <= 1e-12:
        sigma = float(np.std(values))
    return sigma


def estimate_quantization_step(values: np.ndarray) -> float | None:
    """Estimate the sensor's smallest repeated measurement increment."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return None

    unique = np.unique(np.round(values, 8))
    if len(unique) < 2:
        return None

    diffs = np.diff(unique)
    diffs = diffs[diffs > 1e-9]
    if len(diffs) == 0:
        return None

    # Use a low percentile rather than the absolute minimum to reduce the
    # chance that a floating-point artifact becomes the estimated resolution.
    return float(np.percentile(diffs, 10))


def estimate_noise_profile(
    raw: np.ndarray,
    clean: np.ndarray,
    quantization_step: float | None = None,
) -> NoiseProfile:
    """Estimate noise from residual = real measurement - clean reference."""

    raw = np.asarray(raw, dtype=float)
    clean = np.asarray(clean, dtype=float)
    residual = raw - clean

    sigma = robust_sigma(residual)
    if sigma <= 1e-12:
        sigma = max(float(np.std(residual)), 1e-12)

    # Winsorize extreme residuals before estimating lag-1 correlation so a few
    # large spikes do not dominate the autocorrelation estimate.
    clipped = np.clip(residual, -4.0 * sigma, 4.0 * sigma)
    if len(clipped) > 2 and np.std(clipped[:-1]) > 0 and np.std(clipped[1:]) > 0:
        rho = float(np.corrcoef(clipped[:-1], clipped[1:])[0, 1])
        rho = float(np.clip(rho, -0.8, 0.8))
    else:
        rho = 0.0

    spike_mask = np.abs(residual) > 4.0 * sigma
    spike_probability = float(np.mean(spike_mask))
    spike_scale = (
        float(np.median(np.abs(residual[spike_mask])))
        if np.any(spike_mask)
        else 4.0 * sigma
    )

    return NoiseProfile(
        sigma=float(sigma),
        rho=rho,
        spike_probability=spike_probability,
        spike_scale=spike_scale,
        quantization_step=(
            quantization_step
            if quantization_step is not None
            else estimate_quantization_step(raw)
        ),
    )


def add_artificial_noise(
    clean: np.ndarray,
    profile: NoiseProfile,
    level: float,
    seed: int,
    add_spikes: bool = False,
    apply_quantization: bool = True,
) -> np.ndarray:
    """Create one reproducible noisy realization from a clean signal.

    The base noise is AR(1)-correlated Gaussian noise. If rho is near zero this
    reduces to ordinary independent Gaussian noise.
    """

    clean = np.asarray(clean, dtype=float)
    rng = np.random.default_rng(seed)

    sigma = max(profile.sigma * level, 1e-12)
    rho = float(np.clip(profile.rho, -0.8, 0.8))

    # Innovation variance chosen so stationary AR(1) noise has approximately
    # the desired overall standard deviation sigma.
    innovation_sigma = sigma * np.sqrt(max(1e-12, 1.0 - rho**2))
    noise = np.zeros(len(clean), dtype=float)
    if len(noise) > 0:
        noise[0] = rng.normal(0.0, sigma)
    for i in range(1, len(noise)):
        noise[i] = rho * noise[i - 1] + rng.normal(0.0, innovation_sigma)

    noisy = clean + noise

    # Optional robustness scenario for occasional sensor spikes.
    if add_spikes and profile.spike_probability > 0:
        spike_mask = rng.random(len(clean)) < profile.spike_probability
        directions = rng.choice([-1.0, 1.0], size=len(clean))
        magnitudes = np.abs(
            rng.normal(loc=profile.spike_scale, scale=0.25 * profile.spike_scale, size=len(clean))
        )
        noisy = noisy + spike_mask * directions * magnitudes

    # Simulate finite sensor resolution, e.g. 0.01 pH or 0.01 barg.
    if apply_quantization and profile.quantization_step is not None:
        step = profile.quantization_step
        if step > 0:
            noisy = np.round(noisy / step) * step

    return noisy
