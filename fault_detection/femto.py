from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

from fault_detection.pipeline import _window_features

SAMPLE_RATE_HZ = 25_600
CAPTURE_INTERVAL_SECONDS = 10
CAPTURE_SAMPLES = 2_560
SENSOR_COLUMNS = ["horizontal_acceleration", "vertical_acceleration"]
BEARING_PATH = re.compile(r"(?:^|/)(Bearing[^/]+)/acc_(\d+)\.csv$", re.IGNORECASE)


def _read_split(archive: zipfile.ZipFile, split_name: str) -> pd.DataFrame:
    split_info = next(
        (item for item in archive.infolist() if item.filename == split_name), None
    )
    if split_info is None:
        raise ValueError(f"FEMTO archive is missing {split_name}")

    with archive.open(split_info) as stream, zipfile.ZipFile(stream) as split:
        captures: dict[str, list[tuple[int, zipfile.ZipInfo]]] = defaultdict(list)
        for item in split.infolist():
            match = BEARING_PATH.search(item.filename)
            if match:
                captures[match.group(1)].append((int(match.group(2)), item))
        if not captures:
            raise ValueError(f"No bearing capture files found in {split_name}")

        rows: list[dict[str, Any]] = []
        for asset_id, files in sorted(captures.items()):
            files.sort(key=lambda item: item[0])
            indices = [index for index, _ in files]
            if indices != list(range(1, len(indices) + 1)):
                raise ValueError(f"{asset_id} capture numbering is not contiguous")
            for capture_index, info in files:
                with split.open(info) as capture:
                    first_line = capture.readline()
                    delimiter = (
                        ";" if first_line.count(b";") > first_line.count(b",") else ","
                    )
                    capture.seek(0)
                    values = np.loadtxt(
                        capture, delimiter=delimiter, dtype=np.float32
                    )
                if values.shape != (CAPTURE_SAMPLES, 6):
                    raise ValueError(
                        f"{info.filename} has shape {values.shape}; expected "
                        f"({CAPTURE_SAMPLES}, 6)"
                    )
                if not np.isfinite(values[:, 4:6]).all():
                    raise ValueError(f"{info.filename} contains invalid acceleration values")
                rows.append(
                    {
                        **_window_features(
                            values[:, 4:6], sample_rate=SAMPLE_RATE_HZ
                        ),
                        "asset_id": asset_id,
                        "condition_group": asset_id.split("_", maxsplit=1)[0],
                        "capture_index": capture_index,
                        "capture_time_seconds": (
                            capture_index - 1
                        )
                        * CAPTURE_INTERVAL_SECONDS,
                        "rul_seconds": (
                            len(files) - capture_index
                        )
                        * CAPTURE_INTERVAL_SECONDS,
                    }
                )
    return pd.DataFrame(rows)


def _model(random_state: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )


def _summarize(
    truth: np.ndarray, predicted: np.ndarray, assets: np.ndarray
) -> dict[str, Any]:
    per_bearing = {
        str(asset): float(
            mean_absolute_error(truth[assets == asset], predicted[assets == asset])
        )
        for asset in sorted(np.unique(assets))
    }
    return {
        "pooled_mae_seconds": float(mean_absolute_error(truth, predicted)),
        "mean_per_bearing_mae_seconds": float(np.mean(list(per_bearing.values()))),
        "per_bearing_mae_seconds": per_bearing,
    }


def train_femto(
    archive_path: Path, output_dir: Path, predictions_path: Path
) -> dict[str, Any]:
    with zipfile.ZipFile(archive_path) as outer:
        nested_info = next(
            (item for item in outer.infolist() if item.filename.lower().endswith(".zip")),
            None,
        )
        if nested_info is None:
            raise ValueError(f"No nested dataset ZIP found in {archive_path}")
        with outer.open(nested_info) as stream, zipfile.ZipFile(stream) as dataset:
            training = _read_split(dataset, "Training_set.zip")
            validation = _read_split(dataset, "Validation_Set.zip")

    features = [
        column
        for column in training.columns
        if column
        not in {
            "asset_id",
            "condition_group",
            "capture_index",
            "capture_time_seconds",
            "rul_seconds",
        }
    ]
    if features != [
        column
        for column in validation.columns
        if column
        not in {
            "asset_id",
            "condition_group",
            "capture_index",
            "capture_time_seconds",
            "rul_seconds",
        }
    ]:
        raise ValueError("Training and validation feature schemas do not match")

    x_train = training[features]
    y_train = training["rul_seconds"].to_numpy(dtype=float)
    bearing_groups = training["condition_group"].to_numpy()
    cv_predictions = np.empty(len(training), dtype=float)
    cv_baseline = np.empty(len(training), dtype=float)
    splitter = GroupKFold(n_splits=len(np.unique(bearing_groups)))
    for train_indices, test_indices in splitter.split(
        x_train, y_train, groups=bearing_groups
    ):
        regressor = _model(random_state=42)
        regressor.fit(x_train.iloc[train_indices], y_train[train_indices])
        cv_predictions[test_indices] = regressor.predict(x_train.iloc[test_indices])
        cv_baseline[test_indices] = float(np.median(y_train[train_indices]))

    regressor = _model(random_state=42)
    regressor.fit(x_train, y_train)

    x_validation = validation[features]
    validation_truth = validation["rul_seconds"].to_numpy(dtype=float)
    validation_assets = validation["asset_id"].to_numpy()
    validation_predictions = np.maximum(0, regressor.predict(x_validation))
    validation_baseline = np.full(
        len(validation), float(np.median(y_train)), dtype=float
    )

    metrics: dict[str, Any] = {
        "dataset": "NASA FEMTO / PRONOSTIA (PHM 2012)",
        "source_url": "https://phm-datasets.s3.amazonaws.com/NASA/10.+FEMTO+Bearing.zip",
        "target_definition": (
            "Seconds to the final recorded capture in each complete run; captures "
            "are 10 seconds apart. This is a dataset endpoint target, not a field "
            "failure-time guarantee."
        ),
        "signal_definition": (
            "Last two columns of each 2,560-row capture; 25.6 kHz vibration "
            "channels. Features use the shared statistical and spectral window extractor."
        ),
        "algorithm": "RandomForestRegressor(n_estimators=200, min_samples_leaf=2)",
        "training": {
            "bearings": int(training["asset_id"].nunique()),
            "captures": int(len(training)),
        },
        "condition_grouped_cross_validation": {
            "method": "3-fold GroupKFold; bearing operating-condition groups held out",
            "model": _summarize(
                y_train, cv_predictions, training["asset_id"].to_numpy()
            ),
            "constant_median_baseline": _summarize(
                y_train, cv_baseline, training["asset_id"].to_numpy()
            ),
        },
        "full_test_set_holdout": {
            "method": (
                "Full_Test_Set trajectories held out by bearing; target is time to "
                "the final recorded capture."
            ),
            "bearings": int(validation["asset_id"].nunique()),
            "captures": int(len(validation)),
            "model": _summarize(
                validation_truth, validation_predictions, validation_assets
            ),
            "constant_median_baseline": _summarize(
                validation_truth, validation_baseline, validation_assets
            ),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "features": features,
        "sensor_columns": SENSOR_COLUMNS,
        "classifier": None,
        "anomaly_detector": None,
        "rul_regressor": regressor,
        "window_size": CAPTURE_SAMPLES,
        "stride": CAPTURE_SAMPLES,
        "sample_rate": SAMPLE_RATE_HZ,
    }
    joblib.dump(bundle, output_dir / "model.joblib", compress=3)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "asset_id": validation_assets,
            "capture_index": validation["capture_index"],
            "capture_time_seconds": validation["capture_time_seconds"],
            "actual_rul_seconds": validation_truth,
            "predicted_rul_seconds": validation_predictions,
            "baseline_rul_seconds": validation_baseline,
        }
    ).to_csv(predictions_path, index=False)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the NASA FEMTO/PRONOSTIA RUL model"
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("models/femto_rul"))
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("reports/femto_rul_predictions.csv"),
    )
    args = parser.parse_args()
    metrics = train_femto(args.archive, args.output_dir, args.predictions)
    print(json.dumps(metrics, indent=2))
    print(f"Saved model -> {args.output_dir / 'model.joblib'}")


if __name__ == "__main__":
    main()
