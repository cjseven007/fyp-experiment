from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.interpolate import PchipInterpolator


@dataclass
class CleanSignalResult:
    """Stores the automatic clean reference generated from one real signal."""

    clean: np.ndarray
    anchor_indices: np.ndarray
    anchor_values: np.ndarray


class RobustAnchorCleanSignalGenerator:
    """Automatically creates a clean, shape-preserving reference signal.

    Why this approach?
    ------------------
    - It does not require manually drawing a curve by eye.
    - It does not use EMA, SG, weighted SG, or Kalman to create the benchmark,
      avoiding direct circular bias toward one of the candidate filters.
    - Robust local medians suppress short-lived measurement fluctuations.
    - PCHIP interpolation creates a smooth mathematical signal through those
      robust anchors while avoiding the overshoot common with cubic splines.

    Important scientific wording:
    -----------------------------
    With only real noisy SCADA data, this is a *data-informed synthetic
    reference* or *pseudo-ground-truth*, not a physically measured noise-free
    truth. Its reliability improves when several independent reactor cycles are
    included and filter conclusions are averaged across them.
    """

    def __init__(self, n_anchors: int = 41, anchor_half_window: int = 10):
        if n_anchors < 4:
            raise ValueError("n_anchors must be at least 4")
        self.n_anchors = n_anchors
        self.anchor_half_window = max(1, anchor_half_window)

    def generate(self, values: np.ndarray) -> CleanSignalResult:
        values = np.asarray(values, dtype=float)
        n = len(values)

        if n < 10:
            raise ValueError("Signal is too short to construct a clean reference.")

        # Choose anchor locations uniformly across the ENTIRE cycle.
        # Uniform positions make this deterministic and remove visual/manual
        # decisions about where the signal 'looks important'.
        n_anchors = min(self.n_anchors, n)
        anchor_indices = np.unique(np.linspace(0, n - 1, n_anchors).round().astype(int))

        anchor_values = []
        for idx in anchor_indices:
            left = max(0, idx - self.anchor_half_window)
            right = min(n, idx + self.anchor_half_window + 1)

            # Median is robust to spikes/outliers in the local neighborhood.
            local_value = float(np.nanmedian(values[left:right]))
            anchor_values.append(local_value)

        anchor_values = np.asarray(anchor_values, dtype=float)

        # PCHIP = shape-preserving piecewise cubic interpolation. Unlike a
        # regular cubic spline, it is much less likely to create artificial
        # oscillations or overshoot between anchor points.
        interpolator = PchipInterpolator(anchor_indices, anchor_values, extrapolate=True)
        clean = interpolator(np.arange(n, dtype=float))

        return CleanSignalResult(
            clean=np.asarray(clean, dtype=float),
            anchor_indices=anchor_indices,
            anchor_values=anchor_values,
        )
