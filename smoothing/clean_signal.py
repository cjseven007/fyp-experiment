from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator


@dataclass
class CleanSignalResult:
    """Result of automatic clean-reference construction."""

    clean: np.ndarray

    anchor_indices: np.ndarray
    anchor_values: np.ndarray

    # Metadata saved so the reference-generation procedure
    # is fully reproducible.
    spacing_samples: int
    median_window_samples: int
    actual_spacing_seconds: float


class RobustAnchorCleanSignalGenerator:
    """Construct a data-informed clean reference using robust block medians.

    Method
    ------
    1. Anchors are placed at a FIXED PHYSICAL TIME interval.
    2. Each anchor represents the median measurement around that time.
    3. PCHIP connects the robust anchor points.

    The reference therefore captures slower process behaviour while
    suppressing short-term fluctuations.

    Important:
    This is still a pseudo-ground-truth/reference signal because a
    physically measured noise-free reactor signal is unavailable.
    """

    def __init__(
        self,
        anchor_spacing_seconds: float = 60.0,
    ):
        if anchor_spacing_seconds <= 0:
            raise ValueError("anchor_spacing_seconds must be positive")

        self.anchor_spacing_seconds = float(anchor_spacing_seconds)

    def generate(
        self,
        values: np.ndarray,
        sampling_seconds: float,
    ) -> CleanSignalResult:

        values = np.asarray(
            values,
            dtype=float,
        )

        n = len(values)

        if n < 10:
            raise ValueError("Signal is too short to construct " "a clean reference.")

        if sampling_seconds <= 0:
            raise ValueError("sampling_seconds must be positive")

        # -----------------------------------------------------
        # CONVERT PHYSICAL TIME INTO NUMBER OF SAMPLES
        # -----------------------------------------------------

        spacing_samples = max(
            2,
            int(round(self.anchor_spacing_seconds / sampling_seconds)),
        )

        # -----------------------------------------------------
        # CREATE ANCHOR POSITIONS
        # -----------------------------------------------------
        #
        # Example at 1 Hz with 60-second spacing:
        #
        # 0, 60, 120, 180, ...
        #
        # The final sample is always included.

        anchor_indices = np.arange(
            0,
            n,
            spacing_samples,
            dtype=int,
        )

        if anchor_indices[-1] != n - 1:
            anchor_indices = np.append(
                anchor_indices,
                n - 1,
            )

        # -----------------------------------------------------
        # ROBUST LOCAL MEDIAN
        # -----------------------------------------------------
        #
        # Each anchor summarizes approximately the block of
        # measurements represented by that anchor.
        #
        # Therefore the median window is approximately equal
        # to the anchor spacing.
        #
        # This avoids introducing another arbitrary parameter.

        half_window = max(
            1,
            spacing_samples // 2,
        )

        anchor_values = []

        for idx in anchor_indices:

            left = max(
                0,
                idx - half_window,
            )

            right = min(
                n,
                idx + half_window + 1,
            )

            local_values = values[left:right]

            anchor_value = float(np.nanmedian(local_values))

            anchor_values.append(anchor_value)

        anchor_values = np.asarray(
            anchor_values,
            dtype=float,
        )

        # -----------------------------------------------------
        # PCHIP INTERPOLATION
        # -----------------------------------------------------
        #
        # PCHIP is used instead of an ordinary cubic spline
        # because it is shape preserving and avoids artificial
        # oscillatory overshoot between anchor points.

        interpolator = PchipInterpolator(
            anchor_indices,
            anchor_values,
            extrapolate=True,
        )

        sample_positions = np.arange(
            n,
            dtype=float,
        )

        clean = interpolator(sample_positions)

        median_window_samples = 2 * half_window + 1

        return CleanSignalResult(
            clean=np.asarray(
                clean,
                dtype=float,
            ),
            anchor_indices=anchor_indices,
            anchor_values=anchor_values,
            spacing_samples=spacing_samples,
            median_window_samples=(median_window_samples),
            actual_spacing_seconds=(spacing_samples * sampling_seconds),
        )
