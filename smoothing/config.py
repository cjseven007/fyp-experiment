from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class VariableConfig:
    """Maps a canonical variable name to the column in the SCADA CSV."""

    name: str
    column: str
    unit: str


# Change only this mapping if future CSV files use different column names.
VARIABLES: Dict[str, VariableConfig] = {
    "temperature": VariableConfig("temperature", "T-101[C]", "°C"),
    "pressure": VariableConfig("pressure", "P-101[barg]", "barg"),
    "ph": VariableConfig("ph", "pH-101[-]", "pH"),
    "co2": VariableConfig("co2", "CO2[ppm]", "ppm"),
}


@dataclass
class ExperimentConfig:
    """Central place for experiment settings.

    Start with these modest grids for development. After the pipeline works,
    you can widen them for the final FYP experiment.
    """

    # Clean-template extraction settings.
    n_anchors: int = 41
    anchor_half_window: int = 10

    # Artificial-noise levels relative to the estimated real-data noise.
    noise_levels: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])

    # Different random noise realizations used while tuning.
    tuning_seeds: List[int] = field(default_factory=lambda: list(range(5)))

    # Different seeds reserved for final synthetic validation of the winners.
    validation_seeds: List[int] = field(default_factory=lambda: [100, 101, 102])

    # Hyperparameter search spaces.
    # EMA search space.
    # Keep small spans because EMA may need to remain responsive,
    # but the optimizer will now reject spans that do not actually denoise.
    ema_spans: List[int] = field(
        default_factory=lambda: [
            3,
            5,
            9,
            15,
            21,
            31,
            45,
            61,
        ]
    )

    # Avoid extremely small causal SG windows.
    #
    # A window such as 5 with polynomial order 3 can follow the noisy signal
    # too closely, especially at the right edge of a causal SG filter.
    sg_windows: List[int] = field(
        default_factory=lambda: [
            11,
            15,
            21,
            31,
            41,
            61,
            81,
        ]
    )

    # Polynomial orders 1 and 2 are sufficient for preserving local
    # level/slope/curvature without making the filter excessively flexible.
    sg_polyorders: List[int] = field(
        default_factory=lambda: [
            1,
            2,
        ]
    )

    # Weighted SG should not excessively favour the newest measurement.
    #
    # 1.0 = equal weighting
    # 0.8 = mild recent-sample emphasis
    # 0.6 = moderate recent-sample emphasis
    wsg_oldest_weight_ratios: List[float] = field(
        default_factory=lambda: [
            0.6,
            0.8,
            1.0,
        ]
    )

    # Kalman Q/R ratio. A logarithmic grid is appropriate because useful
    # values often span orders of magnitude.
    kalman_q_over_r: List[float] = field(
        default_factory=lambda: [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    )

    # Optional noise effects. Keep spikes off initially for the primary clean
    # Gaussian/correlated-noise benchmark; enable later as a robustness test.
    add_spikes: bool = False
    apply_quantization: bool = True

    # The first samples of large-window causal filters are a warm-up period.
    # We exclude the same fixed number of samples from every candidate so the
    # comparison is fair and no filter gains an advantage from its own window.
    metric_warmup_samples: int = 61

    # Plotting is part of the offline research workflow only. It does not affect
    # filter selection or the future real-time SCADA filtering implementation.
    create_plots: bool = True
    plot_dpi: int = 160

    # ------------------------------------------------------------
    # FILTER ELIGIBILITY REQUIREMENTS
    # ------------------------------------------------------------
    #
    # A configuration must first demonstrate that it actually denoises
    # the signal before it is allowed to compete based on transition
    # preservation.
    #
    # These are initial engineering thresholds and can later be reported
    # as experiment design parameters.

    minimum_snr_improvement_db: float = 0.5

    # Filtered RMSE must improve by at least 5% compared with the
    # unfiltered artificial noisy signal.
    minimum_rmse_reduction_pct: float = 5.0

    # In stable regions, where the true signal changes slowly, the filter
    # should reduce error by at least 5%.
    minimum_stable_rmse_reduction_pct: float = 5.0

    # Roughness ratio:
    #
    #   filtered roughness / noisy roughness
    #
    # 1.0 = essentially no smoothing
    # 0.85 = at least ~15% reduction in short-term roughness
    maximum_stable_roughness_ratio: float = 0.85
