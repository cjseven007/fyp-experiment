import argparse
import json
from dataclasses import asdict
from pathlib import Path

from characterization.analysis import (
    CharacterizationConfig,
    characterize_segments,
    summarize_by_segment,
)
from characterization.plotting import (
    create_characterization_plots,
)


def main():
    parser = argparse.ArgumentParser(
        description="Characterize local pH trends and residual fluctuations."
    )

    parser.add_argument(
        "--prepared",
        required=True,
        help="Folder containing segment_manifest.csv and segments/.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="New or empty output directory.",
    )

    parser.add_argument(
        "--window-seconds",
        type=float,
        default=60.0,
    )

    parser.add_argument(
        "--expected-dt",
        type=float,
        default=1.0,
    )

    args = parser.parse_args()

    config = CharacterizationConfig(
        window_seconds=args.window_seconds,
        expected_dt_seconds=args.expected_dt,
    )

    output_dir = Path(args.output)

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}\n"
            "Choose a new output directory."
        )

    print("\n[1/3] Analyzing experimental windows...", flush=True)

    statistics, skipped, details = characterize_segments(
        prepared_folder=args.prepared,
        config=config,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(
        output_dir / "characterization_config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(asdict(config), file, indent=2)

    skipped.to_csv(
        output_dir / "skipped_windows.csv",
        index=False,
    )

    if statistics.empty:
        print("\nNo windows satisfied the analysis requirements.")
        print("Inspect skipped_windows.csv.")
        print(f"Output: {output_dir.resolve()}")
        return

    print("\n[2/3] Saving statistics...", flush=True)

    statistics.to_csv(
        output_dir / "window_statistics.csv",
        index=False,
    )

    summary = summarize_by_segment(statistics)

    summary.to_csv(
        output_dir / "segment_characterization.csv",
        index=False,
    )

    print("\n[3/3] Generating diagnostic plots...", flush=True)

    create_characterization_plots(
        statistics=statistics,
        details=details,
        output_dir=output_dir,
        config=config,
    )

    print(f"\nAccepted windows: {len(statistics)}")
    print(f"Skipped windows/segments: {len(skipped)}")

    print("\nSegment summary:")
    print(summary.to_string(index=False))

    print(f"\nResults saved to:\n{output_dir.resolve()}")


if __name__ == "__main__":
    main()
