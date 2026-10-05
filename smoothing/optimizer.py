from __future__ import annotations

from itertools import product
from typing import Dict, Iterable, List

import pandas as pd

from .config import ExperimentConfig


def generate_parameter_grid(config: ExperimentConfig) -> Dict[str, List[dict]]:
    """Generate every valid hyperparameter combination for each filter."""

    grid: Dict[str, List[dict]] = {
        "ema": [{"span": span} for span in config.ema_spans],
        "kalman": [{"q_over_r": q} for q in config.kalman_q_over_r],
        "causal_sg": [],
        "weighted_causal_sg": [],
    }

    for window, polyorder in product(config.sg_windows, config.sg_polyorders):
        if polyorder < window:
            grid["causal_sg"].append({"window_length": window, "polyorder": polyorder})

            for ratio in config.wsg_oldest_weight_ratios:
                grid["weighted_causal_sg"].append(
                    {
                        "window_length": window,
                        "polyorder": polyorder,
                        "oldest_weight_ratio": ratio,
                    }
                )

    return grid


def summarize_candidates(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Average repeated noise realizations for each candidate."""

    group_cols = [
        "variable",
        "filter",
        "params",
    ]

    summary = results.groupby(
        group_cols,
        as_index=False,
    ).agg(
        mean_rmse=(
            "rmse",
            "mean",
        ),
        std_rmse=(
            "rmse",
            "std",
        ),
        mean_input_rmse=(
            "input_rmse",
            "mean",
        ),
        mean_rmse_reduction_pct=(
            "rmse_reduction_pct",
            "mean",
        ),
        mean_input_snr_db=(
            "input_snr_db",
            "mean",
        ),
        mean_output_snr_db=(
            "output_snr_db",
            "mean",
        ),
        mean_snr_improvement_db=(
            "snr_improvement_db",
            "mean",
        ),
        std_snr_improvement_db=(
            "snr_improvement_db",
            "std",
        ),
        # Retained for diagnostic/future BOCPD work.
        # These DO NOT determine the current winner.
        mean_transition_rmse=(
            "transition_rmse",
            "mean",
        ),
        mean_stable_rmse=(
            "stable_rmse",
            "mean",
        ),
        mean_stable_roughness_ratio=(
            "stable_roughness_ratio",
            "mean",
        ),
        n_trials=(
            "rmse",
            "size",
        ),
    )

    summary[
        [
            "std_rmse",
            "std_snr_improvement_db",
        ]
    ] = summary[
        [
            "std_rmse",
            "std_snr_improvement_db",
        ]
    ].fillna(0.0)

    # -------------------------------------------------
    # BASELINE COMPARISON
    # -------------------------------------------------
    #
    # No arbitrary 5%, 10%, 0.5 dB threshold.
    #
    # The filter either mathematically improves over
    # the noisy signal or it does not.

    summary["improves_baseline"] = (
        summary["mean_rmse"] < summary["mean_input_rmse"]
    ) & (summary["mean_snr_improvement_db"] > 0.0)

    return summary


def add_selection_ranks(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Add transparent RMSE/SNR ranks for reporting.

    RMSE is the primary selection metric.

    SNR is retained because it is intuitive to report as
    denoising performance, but for a fixed clean reference it
    is mathematically linked to the same squared error as RMSE.
    """

    parts = []

    for variable, part in summary.groupby("variable"):

        part = part.copy()

        part["rank_rmse"] = part["mean_rmse"].rank(
            method="min",
            ascending=True,
        )

        part["rank_snr"] = part["mean_output_snr_db"].rank(
            method="min",
            ascending=False,
        )

        # This score is provided for visualization/reporting only.
        # Actual winner selection below uses RMSE directly.
        part["selection_rank_score"] = (part["rank_rmse"] + part["rank_snr"]) / 2.0

        parts.append(part)

    return pd.concat(
        parts,
        ignore_index=True,
    )


def select_best_per_filter(
    ranked_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Select the best hyperparameters inside each filter family.

    Primary criterion:
        minimum mean RMSE

    Tie-breaker:
        maximum output SNR
    """

    winners = []

    for (
        variable,
        filter_name,
    ), part in ranked_summary.groupby(
        [
            "variable",
            "filter",
        ]
    ):

        ordered = part.sort_values(
            [
                "mean_rmse",
                "mean_output_snr_db",
            ],
            ascending=[
                True,
                False,
            ],
        )

        winners.append(ordered.iloc[0])

    return pd.DataFrame(winners).reset_index(drop=True)


def select_best_per_variable(
    best_per_filter: pd.DataFrame,
) -> pd.DataFrame:
    """Select the overall best filter for each signal variable."""

    winners = []

    for variable, part in best_per_filter.groupby("variable"):

        ordered = part.sort_values(
            [
                "mean_rmse",
                "mean_output_snr_db",
            ],
            ascending=[
                True,
                False,
            ],
        )

        winner = ordered.iloc[0].copy()

        winner["recommended"] = bool(winner["improves_baseline"])

        if not winner["recommended"]:
            print(
                f"WARNING: Best numerical filter for "
                f"{variable} does not improve over "
                f"the unfiltered noisy baseline."
            )

        winners.append(winner)

    return pd.DataFrame(winners).reset_index(drop=True)
