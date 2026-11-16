from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from fault_detection.pipeline import _window_features

ARCHIVE_LABELS = {
    "normal.zip": "normal",
    "horizontal-misalignment.zip": "horizontal_misalignment",
    "vertical-misalignment.zip": "vertical_misalignment",
    "imbalance.zip": "imbalance",
    "underhang.zip": "underhang_bearing",
    "overhang.zip": "overhang_bearing",
}
CLASS_LABELS = list(ARCHIVE_LABELS.values())
SAMPLE_RATE_HZ = 1_000
SOURCE_SAMPLE_RATE_HZ = 50_000
DECIMATION_FACTOR = 50
WINDOW_SIZE = 1_024
SENSOR_COLUMNS = [
    "underhang_axial",
    "underhang_radial",
    "underhang_tangential",
    "overhang_axial",
    "overhang_radial",
    "overhang_tangential",
]


def _extract_recordings(raw_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for archive_name, label in ARCHIVE_LABELS.items():
        archive_path = raw_dir / archive_name
        if not archive_path.is_file():
            raise FileNotFoundError(f"Missing MAFAULDA archive: {archive_path}")
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                if info.is_dir() or not info.filename.lower().endswith(".csv"):
                    continue
                relative_name = info.filename.replace("\\", "/")
                speed_group = Path(relative_name).stem
                recording_id = f"{label}/{relative_name}"
                with archive.open(info) as capture:
                    acceleration = pd.read_csv(
                        capture,
                        header=None,
                        usecols=range(1, 7),
                        dtype=np.float32,
                    ).to_numpy()
                if acceleration.shape != (250_000, 6):
                    raise ValueError(
                        f"{archive_name}:{info.filename} has shape "
                        f"{acceleration.shape}; expected (250000, 6)"
                    )
                if not np.isfinite(acceleration).all():
                    raise ValueError(
                        f"{archive_name}:{info.filename} contains invalid values"
                    )
                decimated = resample_poly(
                    acceleration, up=1, down=DECIMATION_FACTOR, axis=0
                )
                if decimated.shape != (5_000, 6):
                    raise ValueError(
                        f"Unexpected decimated shape for {archive_name}:{info.filename}: "
                        f"{decimated.shape}"
                    )
                counts[label] += 1
                processed = sum(counts.values())
                if processed % 100 == 0:
                    print(f"MAFAULDA: processed {processed} recordings", flush=True)
                for start in range(0, len(decimated) - WINDOW_SIZE + 1, WINDOW_SIZE):
                    window = decimated[start : start + WINDOW_SIZE]
                    rows.append(
                        {
                            **_window_features(window, sample_rate=SAMPLE_RATE_HZ),
                            "fault_type": label,
                            "recording_id": recording_id,
                            "speed_group": speed_group,
                        }
                    )
        print(
            f"MAFAULDA: finished {counts[label]} recordings from {archive_name}",
            flush=True,
        )
    expected_counts = {
        "normal": 49,
        "horizontal_misalignment": 197,
        "vertical_misalignment": 301,
        "imbalance": 333,
        "underhang_bearing": 558,
        "overhang_bearing": 513,
    }
    if dict(counts) != expected_counts:
        raise ValueError(
            f"Recording counts do not match the source catalogue: {dict(counts)}"
        )
    return pd.DataFrame(rows)


def _classifier(random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )


def _scores(truth: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(
            f1_score(
                truth,
                predicted,
                labels=CLASS_LABELS,
                average="macro",
                zero_division=0,
            )
        ),
    }


def _anomaly_scores(
    truth: np.ndarray, predicted: np.ndarray
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "f1": float(f1_score(truth, predicted, zero_division=0)),
    }


def train_mafaulda(
    raw_dir: Path,
    output_dir: Path,
    predictions_path: Path,
    random_state: int = 42,
) -> dict[str, Any]:
    windows = _extract_recordings(raw_dir)
    features = [
        column
        for column in windows.columns
        if column not in {"fault_type", "recording_id", "speed_group"}
    ]
    x = windows[features]
    labels = windows["fault_type"].to_numpy()
    recording_groups = windows["recording_id"].to_numpy()
    speed_groups = windows["speed_group"].to_numpy()
    unique_labels = np.unique(labels)

    recording_split = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state).split(
            x, labels, groups=recording_groups
        )
    )
    recording_train, recording_test = recording_split
    recording_model = _classifier(random_state)
    recording_model.fit(x.iloc[recording_train], labels[recording_train])
    recording_predictions = recording_model.predict(x.iloc[recording_test])

    speed_predictions = np.empty(len(windows), dtype=object)
    speed_folds = np.empty(len(windows), dtype=int)
    speed_splitter = GroupKFold(n_splits=5)
    speed_fold_scores: list[dict[str, Any]] = []
    speed_baseline_predictions = np.empty(len(windows), dtype=object)
    anomaly_truth = labels != "normal"
    anomaly_predictions = np.empty(len(windows), dtype=bool)
    anomaly_fold_scores: list[dict[str, Any]] = []
    for fold, (train_indices, test_indices) in enumerate(
        speed_splitter.split(x, labels, groups=speed_groups), start=1
    ):
        model = _classifier(random_state)
        model.fit(x.iloc[train_indices], labels[train_indices])
        predicted = model.predict(x.iloc[test_indices])
        speed_predictions[test_indices] = predicted
        speed_folds[test_indices] = fold
        majority_label = Counter(labels[train_indices]).most_common(1)[0][0]
        speed_baseline_predictions[test_indices] = majority_label
        speed_fold_scores.append(
            {
                "fold": fold,
                "held_out_speed_groups": int(
                    np.unique(speed_groups[test_indices]).size
                ),
                "held_out_windows": int(len(test_indices)),
                "training_classes": sorted(np.unique(labels[train_indices]).tolist()),
                "model": _scores(labels[test_indices], predicted),
                "majority_baseline": _scores(
                    labels[test_indices],
                    np.full(len(test_indices), majority_label, dtype=object),
                ),
            }
        )
        normal_train_indices = train_indices[labels[train_indices] == "normal"]
        if len(normal_train_indices) < 2:
            raise ValueError(f"Fold {fold} has fewer than two healthy training windows")
        detector = IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=random_state,
            n_jobs=-1,
        )
        detector.fit(x.iloc[normal_train_indices])
        anomaly_predictions[test_indices] = (
            detector.predict(x.iloc[test_indices]) == -1
        )
        anomaly_fold_scores.append(
            {
                "fold": fold,
                "held_out_windows": int(len(test_indices)),
                "normal_training_windows": int(len(normal_train_indices)),
                **_anomaly_scores(
                    anomaly_truth[test_indices], anomaly_predictions[test_indices]
                ),
            }
        )

    classifier = _classifier(random_state)
    classifier.fit(x, labels)
    anomaly_detector = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    normal_mask = labels == "normal"
    anomaly_detector.fit(x.loc[normal_mask])
    metrics: dict[str, Any] = {
        "dataset": "MaFaulDa Machinery Fault Database",
        "source_url": "https://www02.smt.ufrj.br/~offshore/mfs/page_01.html",
        "classes": CLASS_LABELS,
        "recordings_by_class": {
            label: int(windows.loc[windows["fault_type"] == label, "recording_id"].nunique())
            for label in CLASS_LABELS
        },
        "total_recordings": int(windows["recording_id"].nunique()),
        "total_windows": int(len(windows)),
        "sampling": {
            "original_rate_hz": SOURCE_SAMPLE_RATE_HZ,
            "decimation": f"polyphase anti-aliased {DECIMATION_FACTOR}:1",
            "model_rate_hz": SAMPLE_RATE_HZ,
            "window_size_samples": WINDOW_SIZE,
            "stride_samples": WINDOW_SIZE,
            "sensor_channels": SENSOR_COLUMNS,
            "excluded_columns": ["tachometer", "microphone"],
        },
        "algorithm": (
            "RandomForestClassifier(n_estimators=200, min_samples_leaf=2, "
            "class_weight='balanced_subsample')"
        ),
        "recording_group_holdout": {
            "method": "20% GroupShuffleSplit; all windows from each recording stay together",
            **_scores(labels[recording_test], recording_predictions),
            "test_windows": int(len(recording_test)),
        },
        "exact_speed_grouped_cross_validation": {
            "method": (
                "5-fold GroupKFold; all recordings with the same filename speed "
                "stem are held out together"
            ),
            "folds": speed_fold_scores,
            "out_of_fold_model": _scores(labels, speed_predictions),
            "out_of_fold_majority_baseline": _scores(labels, speed_baseline_predictions),
            "held_out_speed_groups": int(np.unique(speed_groups).size),
            "classes": sorted(unique_labels.tolist()),
            "caveat": (
                "Exact filename-speed groups are disjoint; nearby operating speeds "
                "may still occur in different folds. Laboratory results are not "
                "field validation."
            ),
        },
        "anomaly_detection": {
            "method": (
                "5-fold exact-speed-grouped evaluation; IsolationForest is fit "
                "only on healthy training windows"
            ),
            "folds": anomaly_fold_scores,
            "out_of_fold": _anomaly_scores(anomaly_truth, anomaly_predictions),
            "normal_training_windows": int(normal_mask.sum()),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "features": features,
        "sensor_columns": SENSOR_COLUMNS,
        "classifier": classifier,
        "anomaly_detector": anomaly_detector,
        "rul_regressor": None,
        "window_size": WINDOW_SIZE,
        "stride": WINDOW_SIZE,
        "sample_rate": SAMPLE_RATE_HZ,
    }
    joblib.dump(bundle, output_dir / "model.joblib", compress=3)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "recording_id": windows["recording_id"],
            "speed_group": speed_groups,
            "actual_fault": labels,
            "predicted_fault": speed_predictions,
            "actual_anomaly": anomaly_truth,
            "predicted_anomaly": anomaly_predictions,
            "fold": speed_folds,
        }
    ).to_csv(predictions_path, index=False)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the six-class MaFaulDa model"
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/mafaulda"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("models/mafaulda_six_class")
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("reports/mafaulda_six_class_predictions.csv"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    metrics = train_mafaulda(
        args.raw_dir, args.output_dir, args.predictions, random_state=args.seed
    )
    print(json.dumps(metrics, indent=2))
    print(f"Saved model -> {args.output_dir / 'model.joblib'}")


if __name__ == "__main__":
    main()
