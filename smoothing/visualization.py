from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import ExperimentConfig, VARIABLES
from .filters import build_filter
from .noise import add_artificial_noise

# Human-readable names used only in figures.
FILTER_LABELS = {
    "ema": "EMA",
    "kalman": "Kalman",
    "causal_sg": "Causal SG",
    "weighted_causal_sg": "Weighted Causal SG",
}


def _prepare_output_dir(output_dir: str | Path, subdir: str) -> Path:
    """Create a plot subdirectory and return its Path."""
    path = Path(output_dir) / "plots" / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path


def _finish_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    """Apply final layout, save, and close to avoid memory leaks."""
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _parse_params_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Expand the JSON parameter column into normal DataFrame columns."""
    if df.empty:
        return df.copy()

    expanded = df.copy()
    parsed = expanded["params"].apply(json.loads)
    params_df = pd.json_normalize(parsed)
    return pd.concat(
        [expanded.reset_index(drop=True), params_df.reset_index(drop=True)],
        axis=1,
    )


def configure_plot_style() -> None:
    """Use one consistent, publication-friendly plotting style.

    Seaborn controls typography/grid defaults while all figures are still
    ordinary Matplotlib figures and can be further customized later.
    """
    sns.set_theme(context="notebook", style="whitegrid")


# ---------------------------------------------------------------------------
# 1. RAW DATA VS AUTOMATIC CLEAN REFERENCE
# ---------------------------------------------------------------------------


def plot_clean_references(
    prepared: Iterable[dict],
    output_dir: str | Path,
    dpi: int = 160,
) -> None:
    """Plot measured SCADA data against the generated clean reference.

    These figures are important because the clean reference is the foundation
    of SNR/RMSE calculations. You should audit these plots before interpreting
    the filter-ranking results.
    """
    plot_dir = _prepare_output_dir(output_dir, "01_clean_references")

    for item in prepared:
        variable = item["variable"]
        cfg = VARIABLES[variable]
        sample = np.arange(len(item["raw"]))

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(sample, item["raw"], linewidth=1.0, alpha=0.65, label="Measured SCADA")
        ax.plot(
            sample,
            item["clean"],
            linewidth=2.0,
            label="Data-informed clean reference",
        )

        # Show the robust anchor points that were used by PCHIP.
        ax.scatter(
            item["anchor_indices"],
            item["anchor_values"],
            s=20,
            label="Robust median anchors",
            zorder=3,
        )

        ax.set_title(
            f"{cfg.name.capitalize()}: measured signal vs clean reference\n"
            f"{item['source_file']}"
        )
        ax.set_xlabel("Sample")
        ax.set_ylabel(f"{cfg.name.capitalize()} ({cfg.unit})")
        ax.legend()

        filename = f"dataset{item['dataset_id']}_{variable}_clean_reference.png"
        _finish_figure(fig, plot_dir / filename, dpi)


# ---------------------------------------------------------------------------
# 2. RESIDUAL / ESTIMATED-NOISE DISTRIBUTION
# ---------------------------------------------------------------------------


def plot_residual_distributions(
    prepared: Iterable[dict],
    output_dir: str | Path,
    dpi: int = 160,
) -> None:
    """Plot residual = measured - clean reference for each dataset/variable.

    The residual distribution helps justify the artificial-noise model and
    lets you see whether there are large tails/spikes that are not captured by
    a simple Gaussian assumption.
    """
    plot_dir = _prepare_output_dir(output_dir, "02_noise_residuals")

    for item in prepared:
        variable = item["variable"]
        cfg = VARIABLES[variable]
        residual = np.asarray(item["raw"]) - np.asarray(item["clean"])

        fig, ax = plt.subplots(figsize=(8, 5))

        # Matplotlib histogram is used here because it is lightweight and does
        # not require choosing a custom colour/palette. Density=True makes
        # signals with different sample counts comparable.
        ax.hist(
            residual,
            bins="auto",
            density=True,
            alpha=0.75,
        )
        ax.axvline(0.0, linestyle="--", linewidth=1.0)
        ax.set_title(
            f"{cfg.name.capitalize()}: estimated residual distribution\n"
            f"{item['source_file']}"
        )
        ax.set_xlabel(f"Residual ({cfg.unit})")
        ax.set_ylabel("Density")

        filename = f"dataset{item['dataset_id']}_{variable}_residual_distribution.png"
        _finish_figure(fig, plot_dir / filename, dpi)


# ---------------------------------------------------------------------------
# 3. CLEAN REFERENCE VS ARTIFICIALLY NOISY SIGNAL
# ---------------------------------------------------------------------------


def plot_artificial_noisy_signals(
    prepared: Iterable[dict],
    config: ExperimentConfig,
    output_dir: str | Path,
    dpi: int = 160,
) -> None:
    """Plot the synthetic noisy inputs BEFORE any smoothing filter is applied.

    One figure is generated for every:
        dataset x variable x configured noise level

    This gives a direct visual record of the controlled input presented to the
    filters. A held-out validation seed is used so these figures are not tied
    to the noise realizations used during hyperparameter tuning.
    """
    plot_dir = _prepare_output_dir(output_dir, "03_artificial_noisy_signals")

    # Use a validation seed so the plotted example is independent from the
    # realizations used to tune the filter hyperparameters.
    example_seed = config.validation_seeds[0]

    for item in prepared:
        variable = item["variable"]
        cfg = VARIABLES[variable]
        sample = np.arange(len(item["clean"]))

        for noise_level in config.noise_levels:
            noisy = add_artificial_noise(
                clean=item["clean"],
                profile=item["noise_profile"],
                level=noise_level,
                seed=example_seed,
                add_spikes=config.add_spikes,
                apply_quantization=config.apply_quantization,
            )

            fig, ax = plt.subplots(figsize=(11, 5))

            # The clean reference is the known target used when calculating
            # RMSE/SNR. The noisy signal is what the smoothing filters receive.
            ax.plot(
                sample,
                item["clean"],
                linewidth=2.0,
                label="Clean reference",
            )
            ax.plot(
                sample,
                noisy,
                linewidth=0.9,
                alpha=0.65,
                label=f"Artificial noisy signal ({noise_level}× noise)",
            )

            ax.set_title(
                f"{cfg.name.capitalize()}: artificial-noise generation\n"
                f"{item['source_file']} | noise = {noise_level}× | seed = {example_seed}"
            )
            ax.set_xlabel("Sample")
            ax.set_ylabel(f"{cfg.name.capitalize()} ({cfg.unit})")
            ax.legend()

            filename = (
                f"dataset{item['dataset_id']}_{variable}_"
                f"noise_{str(noise_level).replace('.', 'p')}x.png"
            )
            _finish_figure(fig, plot_dir / filename, dpi)


# ---------------------------------------------------------------------------
# 4. HYPERPARAMETER SEARCH VISUALIZATION
# ---------------------------------------------------------------------------


def _plot_ema_search(
    variable_df: pd.DataFrame,
    variable: str,
    plot_dir: Path,
    dpi: int,
) -> None:
    part = _parse_params_frame(variable_df[variable_df["filter"] == "ema"])
    if part.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=part.sort_values("span"),
        x="span",
        y="selection_rank_score",
        marker="o",
        ax=ax,
    )
    ax.set_title(f"{variable.capitalize()}: EMA hyperparameter search")
    ax.set_xlabel("EMA span")
    ax.set_ylabel("Selection rank score (lower is better)")
    _finish_figure(fig, plot_dir / f"{variable}_ema_search.png", dpi)


def _plot_kalman_search(
    variable_df: pd.DataFrame,
    variable: str,
    plot_dir: Path,
    dpi: int,
) -> None:

    part = _parse_params_frame(variable_df[variable_df["filter"] == "kalman"])

    if part.empty:
        return

    # Make sure both plotting columns are numeric.
    part["q_over_r"] = pd.to_numeric(
        part["q_over_r"],
        errors="coerce",
    )

    part["selection_rank_score"] = pd.to_numeric(
        part["selection_rank_score"],
        errors="coerce",
    )

    # Remove invalid values.
    part = part[
        part["q_over_r"].notna()
        & part["selection_rank_score"].notna()
        & (part["q_over_r"] > 0)
        & np.isfinite(part["selection_rank_score"])
    ].copy()

    if part.empty:
        print(
            f"WARNING: Skipping Kalman hyperparameter plot "
            f"for {variable}: no valid finite data."
        )
        return

    part = part.sort_values("q_over_r")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.lineplot(
        data=part,
        x="q_over_r",
        y="selection_rank_score",
        marker="o",
        ax=ax,
    )

    ax.set_xscale("log")

    ax.set_title(f"{variable.capitalize()}: " "Kalman hyperparameter search")

    ax.set_xlabel("Q/R ratio (log scale)")

    ax.set_ylabel("Selection rank score " "(lower is better)")

    _finish_figure(
        fig,
        plot_dir / f"{variable}_kalman_search.png",
        dpi,
    )


def _plot_sg_search(
    variable_df: pd.DataFrame,
    variable: str,
    plot_dir: Path,
    dpi: int,
) -> None:
    part = _parse_params_frame(variable_df[variable_df["filter"] == "causal_sg"])
    if part.empty:
        return

    pivot = part.pivot_table(
        index="polyorder",
        columns="window_length",
        values="selection_rank_score",
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        ax=ax,
    )
    ax.set_title(f"{variable.capitalize()}: causal SG hyperparameter search")
    ax.set_xlabel("Window length")
    ax.set_ylabel("Polynomial order")
    _finish_figure(fig, plot_dir / f"{variable}_causal_sg_search.png", dpi)


def _plot_weighted_sg_search(
    variable_df: pd.DataFrame,
    variable: str,
    plot_dir: Path,
    dpi: int,
) -> None:
    part = _parse_params_frame(
        variable_df[variable_df["filter"] == "weighted_causal_sg"]
    )
    if part.empty:
        return

    # Weighted SG has three hyperparameters. To avoid an unreadable 3-D plot,
    # create one 2-D heatmap per polynomial order.
    for polyorder, poly_part in part.groupby("polyorder"):
        pivot = poly_part.pivot_table(
            index="oldest_weight_ratio",
            columns="window_length",
            values="selection_rank_score",
            aggfunc="mean",
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".1f",
            ax=ax,
        )
        ax.set_title(
            f"{variable.capitalize()}: weighted causal SG search "
            f"(polynomial order = {int(polyorder)})"
        )
        ax.set_xlabel("Window length")
        ax.set_ylabel("Oldest-sample weight ratio")
        _finish_figure(
            fig,
            plot_dir / f"{variable}_weighted_causal_sg_poly{int(polyorder)}_search.png",
            dpi,
        )


def plot_hyperparameter_searches(
    candidate_summary: pd.DataFrame,
    output_dir: str | Path,
    dpi: int = 160,
) -> None:
    """Visualize the tuning landscape for every filter and variable."""
    plot_dir = _prepare_output_dir(output_dir, "04_hyperparameter_search")

    for variable, variable_df in candidate_summary.groupby("variable"):
        _plot_ema_search(variable_df, variable, plot_dir, dpi)
        _plot_kalman_search(variable_df, variable, plot_dir, dpi)
        _plot_sg_search(variable_df, variable, plot_dir, dpi)
        _plot_weighted_sg_search(variable_df, variable, plot_dir, dpi)


# ---------------------------------------------------------------------------
# 5. OPTIMIZED FILTER COMPARISON ON HELD-OUT VALIDATION SIGNALS
# ---------------------------------------------------------------------------


def plot_validation_filter_comparison(
    validation_results: pd.DataFrame,
    output_dir: str | Path,
    dpi: int = 160,
) -> None:
    """Compare the tuned winner from each filter family.

    Separate figures are generated for RMSE, SNR improvement, and transition
    RMSE because they have different units/scales.
    """
    plot_dir = _prepare_output_dir(output_dir, "05_filter_comparison")

    metrics = {
        "rmse": ("RMSE", "RMSE"),
        "snr_improvement_db": ("SNR improvement", "SNR improvement (dB)"),
        "transition_rmse": ("Transition-region RMSE", "Transition RMSE"),
        "stable_rmse": ("Stable-region RMSE", "Stable-region RMSE"),
    }

    data = validation_results.copy()
    data["filter_label"] = data["filter"].map(FILTER_LABELS).fillna(data["filter"])

    for variable, part in data.groupby("variable"):
        for metric, (title_metric, ylabel) in metrics.items():
            fig, ax = plt.subplots(figsize=(8.5, 5))

            # Aggregate manually so the exact meaning of the error bar is
            # explicit: mean +/- one standard deviation across held-out trials.
            summary = (
                part.groupby("filter_label", as_index=False)[metric]
                .agg(["mean", "std"])
                .reset_index()
            )
            summary["std"] = summary["std"].fillna(0.0)

            x = np.arange(len(summary))
            ax.bar(
                x,
                summary["mean"],
                yerr=summary["std"],
                capsize=5,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(summary["filter_label"], rotation=15)

            ax.set_title(
                f"{variable.capitalize()}: optimized-filter comparison\n"
                f"{title_metric} on held-out synthetic validation"
            )
            ax.set_xlabel("Filter")
            ax.set_ylabel(ylabel)

            _finish_figure(
                fig,
                plot_dir / f"{variable}_{metric}_filter_comparison.png",
                dpi,
            )


# ---------------------------------------------------------------------------
# 6. ROBUSTNESS ACROSS LOW / NORMAL / HIGH NOISE
# ---------------------------------------------------------------------------


def plot_noise_level_robustness(
    validation_results: pd.DataFrame,
    output_dir: str | Path,
    dpi: int = 160,
) -> None:
    """Show whether filter rankings remain stable as noise severity increases."""
    plot_dir = _prepare_output_dir(output_dir, "06_noise_robustness")

    data = validation_results.copy()
    data["filter_label"] = data["filter"].map(FILTER_LABELS).fillna(data["filter"])

    metrics = {
        "rmse": "RMSE",
        "snr_improvement_db": "SNR improvement (dB)",
        "transition_rmse": "Transition RMSE",
    }

    for variable, part in data.groupby("variable"):
        for metric, ylabel in metrics.items():
            fig, ax = plt.subplots(figsize=(8.5, 5))
            sns.lineplot(
                data=part,
                x="noise_level",
                y=metric,
                hue="filter_label",
                marker="o",
                errorbar="sd",
                ax=ax,
            )
            ax.set_title(
                f"{variable.capitalize()}: robustness across artificial-noise levels"
            )
            ax.set_xlabel("Noise multiplier relative to estimated real noise")
            ax.set_ylabel(ylabel)
            ax.legend(title="Filter")

            _finish_figure(
                fig,
                plot_dir / f"{variable}_{metric}_vs_noise_level.png",
                dpi,
            )


# ---------------------------------------------------------------------------
# 7. FINAL VISUAL CHECK: CLEAN VS NOISY VS SELECTED FILTER
# ---------------------------------------------------------------------------


def plot_selected_filter_examples(
    prepared: Iterable[dict],
    best_per_variable: pd.DataFrame,
    config: ExperimentConfig,
    output_dir: str | Path,
    dpi: int = 160,
) -> None:
    """Plot one normal-noise held-out example for each real dataset/variable.

    This is not used to choose the filter. Selection has already happened using
    quantitative metrics. The figure is only a visual sanity check and a useful
    dissertation illustration.
    """
    plot_dir = _prepare_output_dir(output_dir, "07_selected_filter_examples")

    # Prefer the nominal 1x noise condition if it exists.
    if 1.0 in config.noise_levels:
        example_noise_level = 1.0
    else:
        example_noise_level = config.noise_levels[len(config.noise_levels) // 2]

    # Use a held-out seed rather than a tuning seed.
    example_seed = config.validation_seeds[0]

    winner_lookup = {row["variable"]: row for _, row in best_per_variable.iterrows()}

    for item in prepared:
        variable = item["variable"]
        if variable not in winner_lookup:
            continue

        winner = winner_lookup[variable]
        params = json.loads(winner["params"])

        noisy = add_artificial_noise(
            clean=item["clean"],
            profile=item["noise_profile"],
            level=example_noise_level,
            seed=example_seed,
            add_spikes=config.add_spikes,
            apply_quantization=config.apply_quantization,
        )

        selected_filter = build_filter(winner["filter"], params)
        filtered = selected_filter.apply_array(noisy)

        cfg = VARIABLES[variable]
        sample = np.arange(len(item["clean"]))

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(
            sample, noisy, linewidth=0.9, alpha=0.55, label="Artificial noisy signal"
        )
        ax.plot(sample, item["clean"], linewidth=2.0, label="Clean reference")
        ax.plot(
            sample,
            filtered,
            linewidth=1.5,
            label=f"Selected: {FILTER_LABELS.get(winner['filter'], winner['filter'])}",
        )

        ax.set_title(
            f"{cfg.name.capitalize()}: selected-filter example\n"
            f"noise = {example_noise_level}×, held-out seed = {example_seed}, "
            f"params = {params}"
        )
        ax.set_xlabel("Sample")
        ax.set_ylabel(f"{cfg.name.capitalize()} ({cfg.unit})")
        ax.legend()

        filename = f"dataset{item['dataset_id']}_{variable}_selected_filter_example.png"
        _finish_figure(fig, plot_dir / filename, dpi)


def create_all_plots(
    prepared: Iterable[dict],
    candidate_summary: pd.DataFrame,
    validation_results: pd.DataFrame,
    best_per_variable: pd.DataFrame,
    config: ExperimentConfig,
    output_dir: str | Path,
) -> None:
    """Generate all figures used to inspect and report the smoothing experiment."""

    configure_plot_style()

    plot_clean_references(
        prepared,
        output_dir,
        dpi=config.plot_dpi,
    )
    plot_residual_distributions(
        prepared,
        output_dir,
        dpi=config.plot_dpi,
    )
    plot_artificial_noisy_signals(
        prepared,
        config,
        output_dir,
        dpi=config.plot_dpi,
    )
    plot_hyperparameter_searches(
        candidate_summary,
        output_dir,
        dpi=config.plot_dpi,
    )
    plot_validation_filter_comparison(
        validation_results,
        output_dir,
        dpi=config.plot_dpi,
    )
    plot_noise_level_robustness(
        validation_results,
        output_dir,
        dpi=config.plot_dpi,
    )
    plot_selected_filter_examples(
        prepared,
        best_per_variable,
        config,
        output_dir,
        dpi=config.plot_dpi,
    )
