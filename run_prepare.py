import argparse
from pathlib import Path

from preparation.segments import (
    PreparationConfig,
    prepare_segments,
)
from preparation.plotting import create_preparation_plots


def main():
    parser = argparse.ArgumentParser(
        description="Prepare usable segments and generate visualizations."
    )

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument(
        "--channels",
        nargs="+",
        choices=["co2", "pressure", "temperature", "ph"],
        default=["ph"],
    )

    parser.add_argument(
        "--min-duration",
        type=float,
        default=120.0,
    )

    parser.add_argument(
        "--min-samples",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--max-gap",
        type=float,
        default=3.0,
    )

    args = parser.parse_args()
    output_dir = Path(args.output)

    config = PreparationConfig(
        channels=tuple(args.channels),
        max_gap_seconds=args.max_gap,
        min_duration_seconds=args.min_duration,
        min_samples=args.min_samples,
    )

    print("\n[1/2] Preparing usable segments...", flush=True)

    prepared, manifest = prepare_segments(
        input_path=args.input,
        output_dir=output_dir,
        config=config,
    )

    print("\n[2/2] Generating preparation plots...", flush=True)

    create_preparation_plots(
        prepared=prepared,
        manifest=manifest,
        output_dir=output_dir,
        channels=config.channels,
        max_gap_seconds=config.max_gap_seconds,
    )

    plot_dir = output_dir / "plots"
    images = sorted(plot_dir.rglob("*.png"))

    if not images:
        raise RuntimeError(
            "No PNG figures were generated. "
            "Check the preceding messages for missing timestamps "
            "or plotting errors."
        )

    print(f"\nCompleted: {len(images)} figures generated.")
    print(f"Open this folder:\n{plot_dir.resolve()}")


if __name__ == "__main__":
    main()
