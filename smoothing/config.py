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

    # ------------------------------------------------------------
    # CLEAN REFERENCE
    # ------------------------------------------------------------
    #
    # PCHIP itself does NOT prescribe an anchor interval.
    # We define the reference in physical time instead of choosing
    # a fixed number of anchors.
    #
    # Default = one robust reference point every 60 seconds.
    #
    # For the final experiment, run sensitivity analyses at
    # 30, 60 and 120 seconds.

    reference_anchor_spacing_seconds: float = 60.0

    # ------------------------------------------------------------
    # SYNTHETIC NOISE
    # ------------------------------------------------------------

    # 1.0 = noise magnitude estimated from the real-data residual.
    #
    # 0.5 and 2.0 are retained for robustness analysis but DO NOT
    # determine the winning filter.
    noise_levels: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])

    # Filter selection is performed at the nominal estimated
    # real-noise level.
    selection_noise_level: float = 1.0

    # Primary controlled benchmark:
    # zero-mean Gaussian noise.
    #
    # Correlated noise can later be enabled as a robustness test.
    use_correlated_noise: bool = False

    # Different random noise realizations used while tuning.
    tuning_seeds: List[int] = field(default_factory=lambda: list(range(5)))

    # Different seeds reserved for final synthetic validation of the winners.
    validation_seeds: List[int] = field(default_factory=lambda: [100, 101, 102])

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
            81,
            101,
        ]
    )

    sg_windows: List[int] = field(
        default_factory=lambda: [
            11,
            15,
            21,
            31,
            41,
            61,
            81,
            101,
        ]
    )

    sg_polyorders: List[int] = field(
        default_factory=lambda: [
            1,
            2,
        ]
    )

    wsg_oldest_weight_ratios: List[float] = field(
        default_factory=lambda: [
            0.5,
            0.7,
            0.9,
        ]
    )

    kalman_q_over_r: List[float] = field(
        default_factory=lambda: [
            1e-5,
            3e-5,
            1e-4,
            3e-4,
            1e-3,
            3e-3,
            1e-2,
            3e-2,
            1e-1,
            3e-1,
            1.0,
        ]
    )

    # Optional noise effects. Keep spikes off initially for the primary clean
    # Gaussian/correlated-noise benchmark; enable later as a robustness test.
    add_spikes: bool = False

    # Do not quantize the primary synthetic benchmark.
    #
    # Quantization can be investigated later as a separate
    # robustness condition.
    apply_quantization: bool = False

    # The first samples of large-window causal filters are a warm-up period.
    # We exclude the same fixed number of samples from every candidate so the
    # comparison is fair and no filter gains an advantage from its own window.
    metric_warmup_samples: int = 0

    # Plotting is part of the offline research workflow only. It does not affect
    # filter selection or the future real-time SCADA filtering implementation.
    create_plots: bool = True
    plot_dpi: int = 160

    # ------------------------------------------------------------
    # REPORT / ILLUSTRATION CHUNKS
    # ------------------------------------------------------------
    #
    # These settings affect figures ONLY.
    # They do NOT participate in filter tuning or selection.

    # Length of each zoomed reporting window.
    report_chunk_seconds: float = 120.0

    # Candidate chunk positions are evaluated every 30 seconds.
    # Smaller stride gives finer placement but does not affect
    # the actual filtering experiment.
    report_chunk_stride_seconds: float = 30.0

    # Number of predefined representative chunk types:
    # stable, representative, transition.
    report_chunk_count: int = 3
