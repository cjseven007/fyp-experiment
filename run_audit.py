import argparse
from pathlib import Path

from audit.config import AuditConfig
from audit.loader import load_recording
from audit.quality import add_quality_flags, make_flag_summary
from audit.plotting import create_audit_plots


def main():
    parser = argparse.ArgumentParser(
        description="Audit one experimental reactor recording."
    )

    parser.add_argument("--input", required=True)
    parser.add_argument("--gas", required=True)
    parser.add_argument("--output", default="results/audit")
    parser.add_argument("--freeze-seconds", type=float, default=60.0)

    args = parser.parse_args()

    if args.freeze_seconds <= 0:
        parser.error("--freeze-seconds must be positive")

    input_path = Path(args.input)
    output_dir = Path(args.output) / input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    config = AuditConfig(freeze_min_seconds=args.freeze_seconds)

    raw = load_recording(input_path)

    if raw.empty:
        raise ValueError("The input CSV contains no data rows.")

    audited = add_quality_flags(raw, config)
    audited["gas_type"] = args.gas

    summary = make_flag_summary(audited)

    audited.to_csv(
        output_dir / "annotated_data.csv",
        index=False,
    )

    summary.to_csv(
        output_dir / "flag_summary.csv",
        index=False,
    )

    create_audit_plots(
        audited,
        output_dir=output_dir,
        context_seconds=60,
        boundary_seconds=60,
    )

    print(f"\nRecording: {input_path.name}")
    print(f"Gas: {args.gas}")
    print(f"Rows: {len(audited):,}")
    print(f"Start: {audited['timestamp'].min()}")
    print(f"End: {audited['timestamp'].max()}")

    print("\nFlag summary:")
    print(summary.to_string(index=False))

    print(f"\nOutputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
