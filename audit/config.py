from dataclasses import dataclass, field

SENSORS = {
    "temperature": "T-101[C]",
    "pressure": "P-101[barg]",
    "ph": "pH-101[-]",
    "co2": "CO2[ppm]",
}

UNITS = {
    "temperature": "°C",
    "pressure": "barg",
    "ph": "pH",
    "co2": "ppm",
}


@dataclass
class AuditConfig:
    co2_upper_limit: float = 2001.0

    # Initial screening rule, not a proven fault-detection threshold.
    freeze_min_seconds: float = 60.0

    # Prevent a long timestamp gap from being counted as a continuous freeze.
    max_contiguous_gap_seconds: float = 3.0

    # Temperature, pressure and pH are checked together for shared freezes.
    # CO2 is excluded because clipping can make it independently constant.
    shared_channels: tuple = ("temperature", "pressure", "ph")

    # Exploratory screening bounds, NOT operating/safety limits.
    # Review these against the instruments and experimental conditions.
    bounds: dict = field(
        default_factory=lambda: {
            "temperature": (0.0, 100.0),
            "pressure": (-0.2, 6.0),
            "ph": (0.0, 14.0),
            "co2": (0.0, 2001.0),
        }
    )
