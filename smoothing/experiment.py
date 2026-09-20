from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .clean_signal import RobustAnchorCleanSignalGenerator
from .config import ExperimentConfig, VARIABLES
from .data_loader import load_all_datasets
from .filters import build_filter
from .metrics import calculate_metrics
from .noise import add_artificial_noise, estimate_noise_profile
from .optimizer import (
    add_selection_ranks,
    generate_parameter_grid,
    select_best_per_filter,
    select_best_per_variable,
    summarize_candidates,
)
from .visualization import create_all_plots


class SmoothingExperiment:
    """Runs the complete offline experiment from CSVs to selected filters."""

    def __init__(self, config: ExperimentConfig | None = None):
        self.config = config or ExperimentConfig()
        self.generator = RobustAnchorCleanSignalGenerator(
            n_anchors=self.config.n_anchors,
            anchor_half_window=self.config.anchor_half_window,
        )
        self.parameter_grid = generate_parameter_grid(self.config)

    def _prepare_references(self, datasets: List[pd.DataFrame]):
        """Build clean references and noise profiles for every dataset/variable."""

        prepared = []
        for dataset_id, df in enumerate(datasets):
            source_file = Path(
                df.attrs.get("source_file", f"dataset_{dataset_id}")
            ).name

            for variable, variable_cfg in VARIABLES.items():
                raw = df[variable_cfg.column].to_numpy(dtype=float)

                clean_result = self.generator.generate(raw)
                quantization_step = df.attrs.get("quantization_steps", {}).get(
                    variable_cfg.column
                )
                noise_profile = estimate_noise_profile(
                    raw,
                    clean_result.clean,
                    quantization_step=quantization_step,
                )

                prepared.append(
                    {
                        "dataset_id": dataset_id,
                        "source_file": source_file,
                        "variable": variable,
                        "raw": raw,
                        "clean": clean_result.clean,
                        "noise_profile": noise_profile,
                        "anchor_indices": clean_result.anchor_indices,
                        "anchor_values": clean_result.anchor_values,
                    }
                )

        return prepared

    def _run_seed_set(self, prepared, seeds: List[int], stage: str) -> pd.DataFrame:
        rows = []
        total_variables = len(prepared)

        for item_index, item in enumerate(prepared, start=1):
            variable = item["variable"]
            clean = item["clean"]
            profile = item["noise_profile"]

            print(
                f"[{stage}] {item_index}/{total_variables}: "
                f"{item['source_file']} - {variable}"
            )

            # IMPORTANT CONTROL: for a fixed dataset/variable/noise level/seed,
            # generate the noisy signal ONCE, then feed that exact same noisy
            # realization to every filter and hyperparameter combination.
            for noise_level in self.config.noise_levels:
                for seed in seeds:
                    noisy = add_artificial_noise(
                        clean=clean,
                        profile=profile,
                        level=noise_level,
                        seed=seed,
                        add_spikes=self.config.add_spikes,
                        apply_quantization=self.config.apply_quantization,
                    )

                    for filter_name, combinations in self.parameter_grid.items():
                        for params in combinations:
                            filter_instance = build_filter(filter_name, params)
                            filtered = filter_instance.apply_array(noisy)

                            metric = calculate_metrics(
                                clean=clean,
                                noisy=noisy,
                                filtered=filtered,
                                warmup_samples=self.config.metric_warmup_samples,
                            )

                            rows.append(
                                {
                                    "stage": stage,
                                    "dataset_id": item["dataset_id"],
                                    "source_file": item["source_file"],
                                    "variable": variable,
                                    "noise_level": noise_level,
                                    "seed": seed,
                                    "filter": filter_name,
                                    # JSON makes dictionary parameters stable and
                                    # groupable inside pandas.
                                    "params": json.dumps(params, sort_keys=True),
                                    "rmse": metric.rmse,
                                    # Baseline error before filtering.
                                    "input_rmse": metric.input_rmse,
                                    # How much RMSE improved compared with no filtering.
                                    "rmse_reduction_pct": (metric.rmse_reduction_pct),
                                    "input_snr_db": (metric.input_snr_db),
                                    "output_snr_db": (metric.output_snr_db),
                                    "snr_improvement_db": (metric.snr_improvement_db),
                                    "transition_rmse": (metric.transition_rmse),
                                    "stable_rmse": (metric.stable_rmse),
                                    "input_stable_rmse": (metric.input_stable_rmse),
                                    "stable_rmse_reduction_pct": (
                                        metric.stable_rmse_reduction_pct
                                    ),
                                    "stable_roughness_ratio": (
                                        metric.stable_roughness_ratio
                                    ),
                                }
                            )

        return pd.DataFrame(rows)

    def _validate_winners(
        self, prepared, best_per_filter: pd.DataFrame
    ) -> pd.DataFrame:
        """Evaluate selected hyperparameters on random seeds never used for tuning."""

        rows = []

        for item in prepared:
            variable = item["variable"]
            clean = item["clean"]
            profile = item["noise_profile"]

            variable_winners = best_per_filter[best_per_filter["variable"] == variable]

            for noise_level in self.config.noise_levels:
                for seed in self.config.validation_seeds:
                    noisy = add_artificial_noise(
                        clean=clean,
                        profile=profile,
                        level=noise_level,
                        seed=seed,
                        add_spikes=self.config.add_spikes,
                        apply_quantization=self.config.apply_quantization,
                    )

                    for _, winner in variable_winners.iterrows():
                        params = json.loads(winner["params"])
                        filter_name = winner["filter"]
                        filtered = build_filter(filter_name, params).apply_array(noisy)
                        metric = calculate_metrics(
                            clean,
                            noisy,
                            filtered,
                            warmup_samples=self.config.metric_warmup_samples,
                        )

                        rows.append(
                            {
                                "stage": "validation",
                                "dataset_id": item["dataset_id"],
                                "source_file": item["source_file"],
                                "variable": variable,
                                "noise_level": noise_level,
                                "seed": seed,
                                "filter": filter_name,
                                "params": winner["params"],
                                "rmse": metric.rmse,
                                "input_rmse": metric.input_rmse,
                                "rmse_reduction_pct": (metric.rmse_reduction_pct),
                                "input_snr_db": (metric.input_snr_db),
                                "output_snr_db": (metric.output_snr_db),
                                "snr_improvement_db": (metric.snr_improvement_db),
                                "transition_rmse": (metric.transition_rmse),
                                "stable_rmse": (metric.stable_rmse),
                                "input_stable_rmse": (metric.input_stable_rmse),
                                "stable_rmse_reduction_pct": (
                                    metric.stable_rmse_reduction_pct
                                ),
                                "stable_roughness_ratio": (
                                    metric.stable_roughness_ratio
                                ),
                            }
                        )

        return pd.DataFrame(rows)

    def run(self, data_path: str | Path, output_dir: str | Path = "results"):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        clean_dir = output_dir / "clean_references"
        clean_dir.mkdir(exist_ok=True)

        datasets = load_all_datasets(data_path)
        print(f"Loaded {len(datasets)} dataset(s).")

        prepared = self._prepare_references(datasets)

        # Save generated clean references and noise characteristics for auditability.
        noise_profile_rows = []
        for item in prepared:
            clean_df = pd.DataFrame(
                {
                    "sample": np.arange(len(item["raw"])),
                    "raw": item["raw"],
                    "clean_reference": item["clean"],
                }
            )
            filename = f"dataset{item['dataset_id']}_{item['variable']}.csv"
            clean_df.to_csv(clean_dir / filename, index=False)

            profile = item["noise_profile"]
            noise_profile_rows.append(
                {
                    "dataset_id": item["dataset_id"],
                    "source_file": item["source_file"],
                    "variable": item["variable"],
                    "sigma": profile.sigma,
                    "rho": profile.rho,
                    "spike_probability": profile.spike_probability,
                    "spike_scale": profile.spike_scale,
                    "quantization_step": profile.quantization_step,
                }
            )

        pd.DataFrame(noise_profile_rows).to_csv(
            output_dir / "estimated_noise_profiles.csv", index=False
        )

        # -------------------- TUNING --------------------
        tuning_results = self._run_seed_set(
            prepared=prepared,
            seeds=self.config.tuning_seeds,
            stage="tuning",
        )
        tuning_results.to_csv(output_dir / "all_tuning_results.csv", index=False)

        candidate_summary = summarize_candidates(tuning_results)
        ranked_summary = add_selection_ranks(
            candidate_summary,
            self.config,
        )
        ranked_summary.to_csv(output_dir / "candidate_summary.csv", index=False)

        # Best hyperparameter set inside EMA, SG, WSG and Kalman separately.
        best_per_filter = select_best_per_filter(ranked_summary)
        best_per_filter.to_csv(output_dir / "best_per_filter.csv", index=False)

        # ---------------- HELD-OUT SYNTHETIC VALIDATION ----------------
        validation_results = self._validate_winners(prepared, best_per_filter)
        validation_results.to_csv(output_dir / "validation_results.csv", index=False)

        validation_summary = summarize_candidates(validation_results)
        validation_ranked = add_selection_ranks(
            validation_summary,
            self.config,
        )
        validation_ranked.to_csv(
            output_dir / "validation_candidate_summary.csv",
            index=False,
        )

        # ---------------------------------------------------------
        # DIAGNOSTIC OUTPUT
        # ---------------------------------------------------------
        #
        # Show why candidate filters passed or failed the minimum
        # smoothing requirements.

        diagnostic_columns = [
            "variable",
            "filter",
            "params",
            "mean_rmse_reduction_pct",
            "mean_snr_improvement_db",
            "mean_stable_rmse_reduction_pct",
            "mean_stable_roughness_ratio",
            "mean_transition_rmse",
            "eligible",
        ]

        print("\nValidation eligibility diagnostics:")

        print(
            validation_ranked[diagnostic_columns]
            .sort_values(
                [
                    "variable",
                    "eligible",
                    "mean_snr_improvement_db",
                ],
                ascending=[
                    True,
                    False,
                    False,
                ],
            )
            .to_string(index=False)
        )
        validation_best_per_filter = select_best_per_filter(validation_ranked)

        # The final overall winner for each variable is chosen based on HELD-OUT
        # validation seeds, not the same random noise used during tuning.
        best_per_variable = select_best_per_variable(validation_best_per_filter)
        best_per_variable.to_csv(output_dir / "best_per_variable.csv", index=False)

        # Figures are generated only AFTER quantitative selection. This prevents
        # visual inspection from influencing which filter is declared the winner.
        if self.config.create_plots:
            print("\nGenerating experiment figures...")
            create_all_plots(
                prepared=prepared,
                candidate_summary=ranked_summary,
                validation_results=validation_results,
                best_per_variable=best_per_variable,
                config=self.config,
                output_dir=output_dir,
            )

        print("\nFinal selected filter per variable (held-out synthetic validation):")
        display_cols = [
            "variable",
            "filter",
            "params",
            "mean_rmse",
            "mean_rmse_reduction_pct",
            "mean_snr_improvement_db",
            "mean_stable_rmse_reduction_pct",
            "mean_stable_roughness_ratio",
            "mean_transition_rmse",
            "recommended",
        ]
        print(best_per_variable[display_cols].to_string(index=False))

        print(f"\nDetailed outputs saved to: {output_dir.resolve()}")

        return {
            "tuning_results": tuning_results,
            "candidate_summary": ranked_summary,
            "best_per_filter": best_per_filter,
            "validation_results": validation_results,
            "best_per_variable": best_per_variable,
        }
