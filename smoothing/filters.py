from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict

import numpy as np


class BaseRealtimeFilter:
    """Common interface used by both offline experiments and future SCADA code.

    Each filter implements update(value), which means the exact same class can
    later be called once per incoming SCADA reading. apply_array() simply loops
    through an offline array for this benchmark.
    """

    def reset(self) -> None:
        raise NotImplementedError

    def update(self, value: float) -> float:
        raise NotImplementedError

    def apply_array(self, values: np.ndarray) -> np.ndarray:
        self.reset()
        return np.asarray([self.update(float(v)) for v in values], dtype=float)


class EMAFilter(BaseRealtimeFilter):
    def __init__(self, span: int):
        if span < 1:
            raise ValueError("EMA span must be >= 1")
        self.span = int(span)
        self.alpha = 2.0 / (self.span + 1.0)
        self.reset()

    def reset(self) -> None:
        self._state = None

    def update(self, value: float) -> float:
        if self._state is None:
            self._state = value
        else:
            self._state = self.alpha * value + (1.0 - self.alpha) * self._state
        return float(self._state)


class KalmanFilter1D(BaseRealtimeFilter):
    """Simple causal scalar Kalman filter using a random-walk process model.

    We fix R=1 and tune Q/R through q_over_r. For this scalar random-walk model,
    the smoothing behavior is governed mainly by this ratio.
    """

    def __init__(self, q_over_r: float):
        if q_over_r <= 0:
            raise ValueError("q_over_r must be positive")
        self.q_over_r = float(q_over_r)
        self.Q = self.q_over_r
        self.R = 1.0
        self.reset()

    def reset(self) -> None:
        self.x = None
        self.P = 1.0

    def update(self, value: float) -> float:
        if self.x is None:
            self.x = value
            return float(self.x)

        # Prediction step for x_t = x_(t-1) + process noise.
        x_pred = self.x
        P_pred = self.P + self.Q

        # Measurement update.
        K = P_pred / (P_pred + self.R)
        self.x = x_pred + K * (value - x_pred)
        self.P = (1.0 - K) * P_pred
        return float(self.x)


def _causal_polynomial_coefficients(
    window_length: int,
    polyorder: int,
    oldest_weight_ratio: float | None = None,
) -> np.ndarray:
    """Return coefficients for one-sided polynomial smoothing.

    A polynomial is fitted to samples [t-w+1, ..., t] and evaluated at t.
    Therefore no future samples are used.
    """

    if window_length < 3:
        raise ValueError("window_length must be >= 3")
    if polyorder < 0 or polyorder >= window_length:
        raise ValueError("polyorder must be smaller than window_length")

    x = np.arange(window_length, dtype=float)
    X = np.vander(x, N=polyorder + 1, increasing=True)

    if oldest_weight_ratio is None:
        weights = np.ones(window_length, dtype=float)
    else:
        if not (0.0 < oldest_weight_ratio <= 1.0):
            raise ValueError("oldest_weight_ratio must be in (0, 1]")

        # Exponential ramp where oldest sample receives the requested ratio and
        # newest sample receives weight 1.0. This is independent of window size.
        age_fraction = (window_length - 1 - x) / max(1, window_length - 1)
        weights = oldest_weight_ratio ** age_fraction

    W = np.diag(weights)
    beta_map = np.linalg.pinv(X.T @ W @ X) @ X.T @ W

    # Evaluate fitted polynomial at the newest x position.
    x_eval = float(window_length - 1)
    eval_vector = np.array([x_eval**power for power in range(polyorder + 1)])
    coefficients = eval_vector @ beta_map
    return coefficients


class CausalSavitzkyGolayFilter(BaseRealtimeFilter):
    """One-sided/causal SG filter suitable for real-time use."""

    def __init__(self, window_length: int, polyorder: int):
        self.window_length = int(window_length)
        self.polyorder = int(polyorder)
        self.coefficients = _causal_polynomial_coefficients(
            self.window_length, self.polyorder
        )
        self.reset()

    def reset(self) -> None:
        self.buffer = deque(maxlen=self.window_length)

    def update(self, value: float) -> float:
        self.buffer.append(value)

        # During warm-up there are not enough samples for the configured
        # polynomial window. Return the current raw value; common warm-up samples
        # are excluded later from ALL candidates' metrics.
        if len(self.buffer) < self.window_length:
            return float(value)

        window = np.asarray(self.buffer, dtype=float)
        return float(np.dot(self.coefficients, window))

    def apply_array(self, values: np.ndarray) -> np.ndarray:
        # Fast offline equivalent of repeatedly calling update(). The filter is
        # still causal: output at t uses only [t-window+1, ..., t].
        values = np.asarray(values, dtype=float)
        output = values.copy()
        if len(values) < self.window_length:
            return output
        valid = np.convolve(values, self.coefficients[::-1], mode="valid")
        output[self.window_length - 1 :] = valid
        return output


class WeightedCausalSavitzkyGolayFilter(BaseRealtimeFilter):
    """Causal weighted SG where recent samples can receive more influence."""

    def __init__(self, window_length: int, polyorder: int, oldest_weight_ratio: float):
        self.window_length = int(window_length)
        self.polyorder = int(polyorder)
        self.oldest_weight_ratio = float(oldest_weight_ratio)
        self.coefficients = _causal_polynomial_coefficients(
            self.window_length,
            self.polyorder,
            oldest_weight_ratio=self.oldest_weight_ratio,
        )
        self.reset()

    def reset(self) -> None:
        self.buffer = deque(maxlen=self.window_length)

    def update(self, value: float) -> float:
        self.buffer.append(value)
        if len(self.buffer) < self.window_length:
            return float(value)
        return float(np.dot(self.coefficients, np.asarray(self.buffer, dtype=float)))

    def apply_array(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        output = values.copy()
        if len(values) < self.window_length:
            return output
        valid = np.convolve(values, self.coefficients[::-1], mode="valid")
        output[self.window_length - 1 :] = valid
        return output


def build_filter(filter_name: str, params: Dict[str, float]) -> BaseRealtimeFilter:
    """Factory used by the optimizer to construct any filter uniformly."""

    if filter_name == "ema":
        return EMAFilter(span=int(params["span"]))

    if filter_name == "kalman":
        return KalmanFilter1D(q_over_r=float(params["q_over_r"]))

    if filter_name == "causal_sg":
        return CausalSavitzkyGolayFilter(
            window_length=int(params["window_length"]),
            polyorder=int(params["polyorder"]),
        )

    if filter_name == "weighted_causal_sg":
        return WeightedCausalSavitzkyGolayFilter(
            window_length=int(params["window_length"]),
            polyorder=int(params["polyorder"]),
            oldest_weight_ratio=float(params["oldest_weight_ratio"]),
        )

    raise ValueError(f"Unknown filter: {filter_name}")
