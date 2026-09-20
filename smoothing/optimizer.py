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


def summarize_candidates(results: pd.DataFrame) -> pd.DataFrame:
    """Average repeated simulations for each variable/filter/parameter set."""

    group_cols = ["variable", "filter", "params"]

    summary = results.groupby(group_cols, as_index=False).agg(
        # Filtered overall RMSE.
        mean_rmse=(
            "rmse",
            "mean",
        ),
        std_rmse=(
            "rmse",
            "std",
        ),
        # Baseline RMSE before filtering.
        mean_input_rmse=(
            "input_rmse",
            "mean",
        ),
        # Percentage RMSE improvement.
        mean_rmse_reduction_pct=(
            "rmse_reduction_pct",
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
        mean_transition_rmse=(
            "transition_rmse",
            "mean",
        ),
        mean_stable_rmse=(
            "stable_rmse",
            "mean",
        ),
        mean_input_stable_rmse=(
            "input_stable_rmse",
            "mean",
        ),
        # Improvement specifically during stable sections.
        mean_stable_rmse_reduction_pct=(
            "stable_rmse_reduction_pct",
            "mean",
        ),
        # < 1 indicates reduced short-term roughness.
        mean_stable_roughness_ratio=(
            "stable_roughness_ratio",
            "mean",
        ),
        n_trials=(
            "rmse",
            "size",
        ),
    )

    # std is NaN if a configuration happened to have only one trial.
    summary[["std_rmse", "std_snr_improvement_db"]] = summary[
        ["std_rmse", "std_snr_improvement_db"]
    ].fillna(0.0)

    return summary


def filter_eligible_candidates(
    summary,
    config,
):

    summary = summary.copy()

    summary["eligible"] = (
        (summary["mean_snr_improvement_db"] >= config.minimum_snr_improvement_db)
        & (summary["mean_rmse_reduction_pct"] >= config.minimum_rmse_reduction_pct)
        & (
            summary["mean_stable_rmse_reduction_pct"]
            >= config.minimum_stable_rmse_reduction_pct
        )
        & (
            summary["mean_stable_roughness_ratio"]
            <= config.maximum_stable_roughness_ratio
        )
    )

    return summary


def add_selection_ranks(
    summary: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    """Apply denoising eligibility requirements, then rank candidates.

    A candidate must first prove that it actually smooths/denoises.
    Transition preservation is considered only after those requirements
    have been satisfied.
    """

    ranked_parts = []

    for variable, part in summary.groupby("variable"):

        part = part.copy()

        # -----------------------------------------------------
        # STEP 1: ELIGIBILITY
        # -----------------------------------------------------
        #
        # These requirements prevent an almost-identity filter
        # from winning simply because it closely follows noisy
        # transitions.

        part["eligible"] = (
            (part["mean_snr_improvement_db"] >= config.minimum_snr_improvement_db)
            & (part["mean_rmse_reduction_pct"] >= config.minimum_rmse_reduction_pct)
            & (
                part["mean_stable_rmse_reduction_pct"]
                >= config.minimum_stable_rmse_reduction_pct
            )
            & (
                part["mean_stable_roughness_ratio"]
                <= config.maximum_stable_roughness_ratio
            )
        )

        # -----------------------------------------------------
        # STEP 2: RANK THE ACTUAL PERFORMANCE
        # -----------------------------------------------------

        part["rank_rmse"] = part["mean_rmse"].rank(
            method="min",
            ascending=True,
        )

        part["rank_snr"] = part["mean_snr_improvement_db"].rank(
            method="min",
            ascending=False,
        )

        part["rank_transition"] = part["mean_transition_rmse"].rank(
            method="min",
            ascending=True,
        )

        part["rank_roughness"] = part["mean_stable_roughness_ratio"].rank(
            method="min",
            ascending=True,
        )

        # Denoising and transition preservation remain balanced,
        # while roughness explicitly ensures that actual smoothing
        # is rewarded.
        part["selection_rank_score"] = (
            part["rank_rmse"]
            + part["rank_snr"]
            + part["rank_transition"]
            + part["rank_roughness"]
        ) / 4.0

        ranked_parts.append(part)

    return pd.concat(
        ranked_parts,
        ignore_index=True,
    )


def select_best_per_filter(ranked_summary: pd.DataFrame) -> pd.DataFrame:
    """Choose the best eligible hyperparameters within each filter family."""

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

        # Only allow configurations that passed the
        # denoising/smoothness eligibility requirements.
        eligible_part = part[part["eligible"]].copy()

        if eligible_part.empty:
            # Keep the best fallback configuration only for reporting.
            #
            # It remains marked eligible=False and therefore should not
            # be allowed to become the final selected filter later.
            fallback = part.sort_values(
                [
                    "mean_snr_improvement_db",
                    "mean_rmse",
                ],
                ascending=[
                    False,
                    True,
                ],
            ).iloc[0]

            winners.append(fallback)

            continue

        eligible_part = eligible_part.sort_values(
            [
                "selection_rank_score",
                "mean_rmse",
                "mean_snr_improvement_db",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )

        winners.append(eligible_part.iloc[0])

    return pd.DataFrame(winners).reset_index(drop=True)


def select_best_per_variable(best_per_filter: pd.DataFrame) -> pd.DataFrame:
    """Compare optimized filter families using only eligible candidates."""

    final_rows = []

    for variable, part in best_per_filter.groupby("variable"):

        part = part.copy()

        # Only filters that actually demonstrated sufficient
        # denoising are allowed into final selection.
        eligible = part[part["eligible"]].copy()

        if eligible.empty:
            print(
                f"WARNING: No filter satisfied the minimum "
                f"denoising requirements for {variable}."
            )

            # Keep the least-bad configuration for diagnostic output,
            # but mark that no acceptable smoothing filter was found.
            fallback = (
                part.sort_values(
                    [
                        "mean_snr_improvement_db",
                        "mean_rmse",
                    ],
                    ascending=[
                        False,
                        True,
                    ],
                )
                .iloc[0]
                .copy()
            )

            fallback["recommended"] = False

            final_rows.append(fallback)

            continue

        eligible["final_rank_rmse"] = eligible["mean_rmse"].rank(
            method="min",
            ascending=True,
        )

        eligible["final_rank_snr"] = eligible["mean_snr_improvement_db"].rank(
            method="min",
            ascending=False,
        )

        eligible["final_rank_transition"] = eligible["mean_transition_rmse"].rank(
            method="min",
            ascending=True,
        )

        eligible["final_rank_roughness"] = eligible["mean_stable_roughness_ratio"].rank(
            method="min",
            ascending=True,
        )

        eligible["final_selection_score"] = (
            eligible["final_rank_rmse"]
            + eligible["final_rank_snr"]
            + eligible["final_rank_transition"]
            + eligible["final_rank_roughness"]
        ) / 4.0

        eligible = eligible.sort_values(
            [
                "final_selection_score",
                "mean_rmse",
                "mean_snr_improvement_db",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )

        winner = eligible.iloc[0].copy()
        winner["recommended"] = True

        final_rows.append(winner)

    return pd.DataFrame(final_rows).reset_index(drop=True)
