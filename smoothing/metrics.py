from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MetricResult:
    rmse: float
    input_rmse: float
    rmse_reduction_pct: float

    input_snr_db: float
    output_snr_db: float
    snr_improvement_db: float

    transition_rmse: float

    stable_rmse: float
    input_stable_rmse: float
    stable_rmse_reduction_pct: float

    stable_roughness_ratio: float


def rmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    return float(np.sqrt(np.mean((reference - estimate) ** 2)))


def snr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    """SNR definition consistent with the equation in your interim report."""

    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)

    signal_power = float(np.sum(reference**2))
    error_power = float(np.sum((reference - estimate) ** 2))

    if error_power <= 1e-24:
        return float("inf")
    if signal_power <= 1e-24:
        return float("-inf")

    return float(10.0 * np.log10(signal_power / error_power))


def automatic_region_masks(clean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Automatically define transition and stable regions from clean slope.

    No region is selected by eye. The highest-gradient 20% is treated as the
    transition-focused region; the lowest-gradient 50% is treated as stable.
    """

    clean = np.asarray(clean, dtype=float)
    gradient = np.abs(np.gradient(clean))

    transition_threshold = np.percentile(gradient, 80)
    stable_threshold = np.percentile(gradient, 50)

    transition_mask = gradient >= transition_threshold
    stable_mask = gradient <= stable_threshold

    # Handle near-constant signals where percentile masks may become all True.
    if transition_mask.sum() < 3:
        transition_mask = np.ones(len(clean), dtype=bool)
    if stable_mask.sum() < 3:
        stable_mask = np.ones(len(clean), dtype=bool)

    return transition_mask, stable_mask


def calculate_metrics(
    clean: np.ndarray,
    noisy: np.ndarray,
    filtered: np.ndarray,
    warmup_samples: int = 0,
) -> MetricResult:
    """Calculate denoising, transition, and smoothness metrics.

    The important difference from the previous version is that the
    filtered signal is explicitly compared with the unfiltered noisy
    baseline.

    This prevents an almost-identity filter from winning merely because
    it preserves transitions very closely.
    """

    clean = np.asarray(clean, dtype=float)
    noisy = np.asarray(noisy, dtype=float)
    filtered = np.asarray(filtered, dtype=float)

    # Ignore the common warm-up period of causal filters.
    start = min(
        max(0, warmup_samples),
        max(0, len(clean) - 1),
    )

    clean_eval = clean[start:]
    noisy_eval = noisy[start:]
    filtered_eval = filtered[start:]

    transition_mask, stable_mask = automatic_region_masks(clean_eval)

    # ---------------------------------------------------------
    # WHOLE-SIGNAL RMSE
    # ---------------------------------------------------------

    # Error before filtering.
    input_rmse = rmse(
        clean_eval,
        noisy_eval,
    )

    # Error after filtering.
    output_rmse = rmse(
        clean_eval,
        filtered_eval,
    )

    # Positive percentage = improvement.
    #
    # Example:
    # input RMSE    = 10
    # filtered RMSE = 8
    #
    # reduction = 20%
    rmse_reduction_pct = 100.0 * (input_rmse - output_rmse) / max(input_rmse, 1e-12)

    # ---------------------------------------------------------
    # SNR
    # ---------------------------------------------------------

    input_snr = snr_db(
        clean_eval,
        noisy_eval,
    )

    output_snr = snr_db(
        clean_eval,
        filtered_eval,
    )

    snr_improvement = output_snr - input_snr

    # ---------------------------------------------------------
    # TRANSITION REGION
    # ---------------------------------------------------------

    transition_error = rmse(
        clean_eval[transition_mask],
        filtered_eval[transition_mask],
    )

    # ---------------------------------------------------------
    # STABLE REGION
    # ---------------------------------------------------------

    input_stable_rmse = rmse(
        clean_eval[stable_mask],
        noisy_eval[stable_mask],
    )

    output_stable_rmse = rmse(
        clean_eval[stable_mask],
        filtered_eval[stable_mask],
    )

    stable_rmse_reduction_pct = (
        100.0 * (input_stable_rmse - output_stable_rmse) / max(input_stable_rmse, 1e-12)
    )

    # ---------------------------------------------------------
    # ROUGHNESS MEASUREMENT
    # ---------------------------------------------------------
    #
    # Roughness is measured from consecutive first differences.
    #
    # If the noisy signal jumps strongly from one sample to the next,
    # its roughness will be high.
    #
    # A smoothing filter should reduce that roughness.

    # Only compare consecutive samples where both samples belong
    # to the stable region.
    stable_pair_mask = stable_mask[:-1] & stable_mask[1:]

    noisy_difference = np.diff(noisy_eval)[stable_pair_mask]

    filtered_difference = np.diff(filtered_eval)[stable_pair_mask]

    # If too few stable pairs somehow exist, fall back to the
    # complete signal differences.
    if len(noisy_difference) < 3:
        noisy_difference = np.diff(noisy_eval)
        filtered_difference = np.diff(filtered_eval)

    noisy_roughness = float(np.sqrt(np.mean(noisy_difference**2)))

    filtered_roughness = float(np.sqrt(np.mean(filtered_difference**2)))

    stable_roughness_ratio = filtered_roughness / max(noisy_roughness, 1e-12)

    return MetricResult(
        rmse=output_rmse,
        input_rmse=input_rmse,
        rmse_reduction_pct=(rmse_reduction_pct),
        input_snr_db=input_snr,
        output_snr_db=output_snr,
        snr_improvement_db=(snr_improvement),
        transition_rmse=(transition_error),
        stable_rmse=(output_stable_rmse),
        input_stable_rmse=(input_stable_rmse),
        stable_rmse_reduction_pct=(stable_rmse_reduction_pct),
        stable_roughness_ratio=(stable_roughness_ratio),
    )
