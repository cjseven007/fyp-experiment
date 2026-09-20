import argparse

from smoothing.config import ExperimentConfig
from smoothing.experiment import SmoothingExperiment


def main():
    parser = argparse.ArgumentParser(
        description="Offline smoothing-filter benchmark for FYP SCADA signals."
    )
    parser.add_argument(
        "--data",
        default="data",
        help="Path to one CSV file OR a directory containing any number of CSV files.",
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Directory where experiment result CSV files will be saved.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip Matplotlib/Seaborn figure generation during quick debugging runs.",
    )
    args = parser.parse_args()

    # All experiment settings/hyperparameter grids live in one configuration
    # object so the scientific protocol is easy to inspect and reproduce.
    config = ExperimentConfig()
    config.create_plots = not args.no_plots

    experiment = SmoothingExperiment(config)
    experiment.run(data_path=args.data, output_dir=args.output)


if __name__ == "__main__":
    main()
