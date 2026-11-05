from __future__ import annotations

import argparse
import json
from pathlib import Path

from fault_detection.pipeline import combine_csv, predict, prepare_csv, train_model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fault-detection",
        description="Unified electrical-machine fault and predictive-maintenance pipeline",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare", help="convert CSV samples to the canonical training schema"
    )
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--dataset", required=True, help="dataset/source identifier")
    prepare.add_argument("--sensor-columns", help="comma-separated numeric signal columns")
    prepare.add_argument("--label-column")
    prepare.add_argument("--asset-column")
    prepare.add_argument("--run-column")
    prepare.add_argument("--time-column")
    prepare.add_argument("--rul-column")
    prepare.add_argument("--fault-label", help="use when all files have the same label")
    prepare.add_argument("--derive-rul", action="store_true")
    prepare.add_argument("--sample-rate", type=float)

    combine = commands.add_parser(
        "combine", help="combine canonical datasets with matching sensor schemas"
    )
    combine.add_argument("--inputs", type=Path, nargs="+", required=True)
    combine.add_argument("--output", type=Path, required=True)

    train = commands.add_parser("train", help="train classification, anomaly, and RUL models")
    train.add_argument("--input", type=Path, required=True, help="canonical CSV")
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--window-size", type=int, default=1024)
    train.add_argument("--stride", type=int, default=512)
    train.add_argument("--sample-rate", type=float, default=1.0)
    train.add_argument("--seed", type=int, default=42)

    score = commands.add_parser("predict", help="score canonical CSV with a trained model")
    score.add_argument("--input", type=Path, required=True, help="canonical CSV")
    score.add_argument("--model-dir", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "prepare":
        count = prepare_csv(
            args.input,
            args.output,
            dataset=args.dataset,
            sensor_columns=args.sensor_columns.split(",") if args.sensor_columns else None,
            label_column=args.label_column,
            asset_column=args.asset_column,
            run_column=args.run_column,
            time_column=args.time_column,
            rul_column=args.rul_column,
            fault_label=args.fault_label,
            derive_rul=args.derive_rul,
            sample_rate=args.sample_rate,
        )
        print(f"Prepared {count:,} sensor samples -> {args.output}")
    elif args.command == "combine":
        count = combine_csv(args.inputs, args.output)
        print(f"Combined {count:,} sensor samples -> {args.output}")
    elif args.command == "train":
        metrics = train_model(
            args.input,
            args.output_dir,
            window_size=args.window_size,
            stride=args.stride,
            sample_rate=args.sample_rate,
            random_state=args.seed,
        )
        print(json.dumps(metrics, indent=2))
        print(f"Saved model -> {args.output_dir / 'model.joblib'}")
    else:
        count = predict(args.input, args.model_dir, args.output)
        print(f"Scored {count:,} windows -> {args.output}")


if __name__ == "__main__":
    main()
