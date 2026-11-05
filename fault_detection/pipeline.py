from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit

CANONICAL_COLUMNS = {
    "dataset",
    "asset_id",
    "run_id",
    "timestamp",
    "fault_type",
    "rul_seconds",
}
NORMAL_LABELS = {"normal", "healthy", "baseline", "no_fault", "no fault"}
ALIASES = {
    "healthy": "normal",
    "baseline": "normal",
    "no_fault": "normal",
    "no fault": "normal",
    "ball": "bearing_fault",
    "inner_race": "bearing_fault",
    "outer_race": "bearing_fault",
    "inner_race_fault": "bearing_fault",
    "outer_race_fault": "bearing_fault",
    "bearing": "bearing_fault",
    "bearing_fault": "bearing_fault",
    "gear": "gear_fault",
    "gear_tooth": "gear_fault",
    "gear_fault": "gear_fault",
    "shaft": "shaft_fault",
    "shaft_fault": "shaft_fault",
    "imbalance": "imbalance",
    "unbalance": "imbalance",
    "misalignment": "misalignment",
    "bowed_shaft": "shaft_fault",
}


def _label(value: Any) -> str:
    text = re.sub(r"[\s-]+", "_", str(value).strip().lower())
    return ALIASES.get(text, text)


def _infer_label(path: Path, root: Path) -> str:
    generic = {"", "data", "dataset", "datasets", "raw", "processed"}
    for parent in path.parents:
        if parent == root.parent:
            break
        if parent.name.lower() not in generic:
            return _label(parent.name)
    raise ValueError(
        f"Cannot infer a fault label for {path}. Pass --fault-label or include a "
        "fault_type column."
    )


def prepare_csv(
    source: Path,
    output: Path,
    *,
    dataset: str,
    sensor_columns: list[str] | None = None,
    label_column: str | None = None,
    asset_column: str | None = None,
    run_column: str | None = None,
    time_column: str | None = None,
    rul_column: str | None = None,
    fault_label: str | None = None,
    derive_rul: bool = False,
    sample_rate: float | None = None,
) -> int:
    files = [source] if source.is_file() else sorted(source.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {source}")

    frames: list[pd.DataFrame] = []
    root = source if source.is_dir() else source.parent
    for path in files:
        raw = pd.read_csv(path)
        if raw.empty:
            continue
        requested_columns = {
            "label": label_column,
            "asset": asset_column,
            "run": run_column,
            "time": time_column,
            "RUL": rul_column,
        }
        for field, column in requested_columns.items():
            if column and column not in raw.columns:
                raise ValueError(f"{path} is missing the configured {field} column {column!r}")
        chosen_sensors = sensor_columns or [
            column
            for column in raw.columns
            if column not in {label_column, asset_column, run_column, time_column, rul_column}
            and pd.api.types.is_numeric_dtype(raw[column])
        ]
        missing = [column for column in chosen_sensors if column not in raw.columns]
        if missing:
            raise ValueError(f"{path} is missing sensor columns: {', '.join(missing)}")
        if not chosen_sensors:
            raise ValueError(f"No numeric sensor columns found in {path}")

        frame = pd.DataFrame(index=raw.index)
        frame["dataset"] = dataset
        frame["asset_id"] = (
            raw[asset_column].astype(str)
            if asset_column
            else str(path.relative_to(root).with_suffix(""))
        )
        frame["run_id"] = (
            raw[run_column].astype(str)
            if run_column
            else "run_0"
        )
        if time_column:
            frame["timestamp"] = pd.to_numeric(raw[time_column], errors="coerce")
            frame["timestamp"] = frame["timestamp"].fillna(
                pd.Series(np.arange(len(raw)), index=raw.index)
            )
        else:
            frame["timestamp"] = np.arange(len(raw))

        if label_column:
            frame["fault_type"] = raw[label_column].map(_label)
        elif fault_label:
            frame["fault_type"] = _label(fault_label)
        else:
            frame["fault_type"] = _infer_label(path, root)
        if rul_column:
            frame["rul_seconds"] = pd.to_numeric(raw[rul_column], errors="coerce")
        else:
            frame["rul_seconds"] = np.nan
        for column in chosen_sensors:
            frame[column] = pd.to_numeric(raw[column], errors="coerce")
        frames.append(frame)

    if not frames:
        raise ValueError(f"No non-empty CSV data found under {source}")

    data = pd.concat(frames, ignore_index=True)
    sensor_names = list(dict.fromkeys(column for frame in frames for column in frame.columns if column not in CANONICAL_COLUMNS))
    data[sensor_names] = data[sensor_names].replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=sensor_names, how="all")
    if data.empty:
        raise ValueError("No rows contain valid numeric sensor measurements")

    if derive_rul:
        if not sample_rate or sample_rate <= 0:
            raise ValueError("--derive-rul requires a positive --sample-rate in Hz")
        keys = ["asset_id", "run_id"]
        for _, indices in data.groupby(keys, sort=False).groups.items():
            ordered_indices = data.loc[indices].sort_values("timestamp").index
            remaining = np.arange(len(ordered_indices) - 1, -1, -1) / sample_rate
            data.loc[ordered_indices, "rul_seconds"] = remaining

    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False)
    return len(data)


def combine_csv(inputs: list[Path], output: Path) -> int:
    if not inputs:
        raise ValueError("Provide at least one prepared CSV")
    frames: list[pd.DataFrame] = []
    expected_sensors: list[str] | None = None
    for path in inputs:
        frame = pd.read_csv(path)
        missing = CANONICAL_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(
                f"{path} is not canonical; missing columns: {', '.join(sorted(missing))}"
            )
        sensors = [column for column in frame.columns if column not in CANONICAL_COLUMNS]
        if expected_sensors is None:
            expected_sensors = sensors
        elif sensors != expected_sensors:
            raise ValueError(
                f"Sensor schema mismatch in {path}. Expected {expected_sensors}, got {sensors}. "
                "Map/resample compatible channels before combining datasets."
            )
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    return len(combined)


def _window_features(window: np.ndarray, sample_rate: float) -> dict[str, float]:
    result: dict[str, float] = {}
    length = len(window)
    for channel_index in range(window.shape[1]):
        values = window[:, channel_index].astype(np.float64, copy=False)
        values = values[np.isfinite(values)]
        if len(values) < 2:
            continue
        prefix = f"ch{channel_index}"
        mean = float(values.mean())
        centered = values - mean
        std = float(values.std())
        rms = float(np.sqrt(np.mean(values * values)))
        absolute = np.abs(values)
        result.update(
            {
                f"{prefix}_mean": mean,
                f"{prefix}_std": std,
                f"{prefix}_rms": rms,
                f"{prefix}_peak_to_peak": float(np.ptp(values)),
                f"{prefix}_abs_mean": float(absolute.mean()),
                f"{prefix}_skew": float(np.mean(centered**3) / (std**3 + 1e-12)),
                f"{prefix}_kurtosis": float(np.mean(centered**4) / (std**4 + 1e-12)),
                f"{prefix}_crest_factor": float(absolute.max() / (rms + 1e-12)),
            }
        )
        spectrum = np.abs(np.fft.rfft(centered))
        frequencies = np.fft.rfftfreq(len(values), d=1.0 / sample_rate)
        power = spectrum**2
        total_power = float(power.sum()) + 1e-12
        dominant = int(np.argmax(spectrum[1:]) + 1) if len(spectrum) > 1 else 0
        result[f"{prefix}_dominant_hz"] = float(frequencies[dominant])
        result[f"{prefix}_spectral_centroid_hz"] = float(
            np.sum(frequencies * spectrum) / (spectrum.sum() + 1e-12)
        )
        for band_index, (low, high) in enumerate(((0.0, 0.1), (0.1, 0.25), (0.25, 0.5))):
            mask = (frequencies >= low * sample_rate) & (frequencies < high * sample_rate)
            result[f"{prefix}_band{band_index}_power_ratio"] = float(
                power[mask].sum() / total_power
            )
    return result


def make_windows(
    data: pd.DataFrame, window_size: int, stride: int, sample_rate: float
) -> tuple[pd.DataFrame, list[str]]:
    sensors = [column for column in data.columns if column not in CANONICAL_COLUMNS]
    if not sensors:
        raise ValueError("Canonical input has no sensor columns")
    rows: list[dict[str, Any]] = []
    for (asset, run), group in data.groupby(["asset_id", "run_id"], sort=False):
        group = group.sort_values("timestamp")
        values = group[sensors].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        labels = group["fault_type"].astype(str).to_numpy()
        rul = pd.to_numeric(group["rul_seconds"], errors="coerce").to_numpy(dtype=float)
        for start in range(0, len(group) - window_size + 1, stride):
            stop = start + window_size
            segment = values[start:stop]
            if np.isnan(segment).all(axis=0).any():
                continue
            segment = np.nan_to_num(segment, nan=0.0, posinf=0.0, neginf=0.0)
            feature_row = _window_features(segment, sample_rate)
            if not feature_row:
                continue
            label_counts = Counter(labels[start:stop])
            row: dict[str, Any] = {
                **feature_row,
                "asset_id": str(asset),
                "run_id": str(run),
                "fault_type": label_counts.most_common(1)[0][0],
                "rul_seconds": float(np.nanmean(rul[start:stop]))
                if np.isfinite(rul[start:stop]).any()
                else np.nan,
                "start_timestamp": group["timestamp"].iloc[start],
            }
            rows.append(row)
    if not rows:
        raise ValueError(
            f"No windows could be created; check that each run has at least "
            f"{window_size} samples."
        )
    windows = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    feature_columns = [
        column
        for column in windows.columns
        if column
        not in {
            "asset_id",
            "run_id",
            "fault_type",
            "rul_seconds",
            "start_timestamp",
        }
    ]
    windows[feature_columns] = windows[feature_columns].fillna(0.0)
    return windows, feature_columns


def _group_split(groups: np.ndarray, random_state: int) -> tuple[np.ndarray, np.ndarray] | None:
    if len(np.unique(groups)) < 2:
        return None
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_indices, test_indices = next(
        splitter.split(np.zeros(len(groups)), groups=groups)
    )
    return train_indices, test_indices


def train_model(
    input_csv: Path,
    output_dir: Path,
    *,
    window_size: int = 1024,
    stride: int = 512,
    sample_rate: float = 1.0,
    random_state: int = 42,
) -> dict[str, Any]:
    if window_size < 2 or stride < 1 or sample_rate <= 0:
        raise ValueError("window-size must be >= 2, stride >= 1, and sample-rate > 0")
    data = pd.read_csv(input_csv)
    missing = CANONICAL_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Canonical CSV is missing columns: {', '.join(sorted(missing))}")
    windows, features = make_windows(data, window_size, stride, sample_rate)
    groups = windows["asset_id"].to_numpy()
    split = _group_split(groups, random_state)
    metrics: dict[str, Any] = {
        "window_count": int(len(windows)),
        "asset_count": int(windows["asset_id"].nunique()),
        "window_size_samples": window_size,
        "stride_samples": stride,
        "sample_rate_hz": sample_rate,
        "split": "asset_group_holdout" if split else "not_available_single_asset",
    }

    bundle: dict[str, Any] = {
        "features": features,
        "sensor_columns": [
            column for column in data.columns if column not in CANONICAL_COLUMNS
        ],
        "classifier": None,
        "anomaly_detector": None,
        "rul_regressor": None,
        "window_size": window_size,
        "stride": stride,
        "sample_rate": sample_rate,
    }
    labels = windows["fault_type"].map(_label)
    if labels.nunique() >= 2:
        classifier = RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
        if split:
            train_indices, test_indices = split
            classifier.fit(windows.iloc[train_indices][features], labels.iloc[train_indices])
            predicted = classifier.predict(windows.iloc[test_indices][features])
            metrics["classification"] = {
                "accuracy": float(accuracy_score(labels.iloc[test_indices], predicted)),
                "macro_f1": float(
                    f1_score(labels.iloc[test_indices], predicted, average="macro", zero_division=0)
                ),
                "test_windows": int(len(test_indices)),
            }
        classifier.fit(windows[features], labels)
        bundle["classifier"] = classifier
    else:
        metrics["classification"] = "skipped: at least two fault classes are required"

    normal_mask = labels.isin(NORMAL_LABELS)
    if normal_mask.sum() >= 2:
        detector = IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=random_state,
            n_jobs=-1,
        )
        detector.fit(windows.loc[normal_mask, features])
        bundle["anomaly_detector"] = detector
        metrics["anomaly_detection"] = {
            "normal_training_windows": int(normal_mask.sum()),
            "method": "IsolationForest trained on normal windows",
        }
    else:
        metrics["anomaly_detection"] = (
            "skipped: include at least two windows labelled normal/healthy"
        )

    rul_values = pd.to_numeric(windows["rul_seconds"], errors="coerce")
    rul_mask = rul_values.notna() & np.isfinite(rul_values)
    if rul_mask.sum() >= 2:
        regressor = RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )
        if split:
            train_indices, test_indices = split
            train_rul = rul_mask.iloc[train_indices].to_numpy()
            test_rul = rul_mask.iloc[test_indices].to_numpy()
            if train_rul.sum() >= 2 and test_rul.sum() > 0:
                regressor.fit(
                    windows.iloc[train_indices].loc[train_rul, features],
                    rul_values.iloc[train_indices][train_rul],
                )
                predicted = regressor.predict(
                    windows.iloc[test_indices].loc[test_rul, features]
                )
                metrics["rul"] = {
                    "mae_seconds": float(
                        mean_absolute_error(rul_values.iloc[test_indices][test_rul], predicted)
                    ),
                    "test_windows": int(test_rul.sum()),
                }
        regressor.fit(windows.loc[rul_mask, features], rul_values.loc[rul_mask])
        bundle["rul_regressor"] = regressor
    else:
        metrics["rul"] = "skipped: provide rul_seconds or use --derive-rul during preparation"

    if all(
        bundle[name] is None
        for name in ("classifier", "anomaly_detector", "rul_regressor")
    ):
        raise ValueError(
            "No model was trained. Provide at least two fault labels, normal windows, "
            "or RUL labels."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_dir / "model.joblib")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


def predict(input_csv: Path, model_dir: Path, output_csv: Path) -> int:
    bundle = joblib.load(model_dir / "model.joblib")
    data = pd.read_csv(input_csv)
    sensor_columns = [
        column for column in data.columns if column not in CANONICAL_COLUMNS
    ]
    if sensor_columns != bundle["sensor_columns"]:
        raise ValueError(
            "Sensor columns do not match the trained model. "
            f"Expected {bundle['sensor_columns']}, got {sensor_columns}."
        )
    windows, features = make_windows(
        data,
        bundle["window_size"],
        bundle["stride"],
        bundle["sample_rate"],
    )
    expected = bundle["features"]
    if features != expected:
        raise ValueError(
            "Sensor columns/features do not match the trained model. "
            "Prepare prediction data with the same sensor columns and order."
        )
    result = windows[["asset_id", "run_id", "start_timestamp"]].copy()
    classifier = bundle["classifier"]
    if classifier is not None:
        probabilities = classifier.predict_proba(windows[features])
        winner = probabilities.argmax(axis=1)
        result["predicted_fault"] = classifier.classes_[winner]
        result["fault_confidence"] = probabilities.max(axis=1)
    detector = bundle["anomaly_detector"]
    if detector is not None:
        result["anomaly"] = detector.predict(windows[features]) == -1
        result["anomaly_score"] = -detector.decision_function(windows[features])
    regressor = bundle["rul_regressor"]
    if regressor is not None:
        result["predicted_rul_seconds"] = np.maximum(
            0, regressor.predict(windows[features])
        )
    if "fault_type" in windows:
        result["actual_fault"] = windows["fault_type"]
    if "rul_seconds" in windows:
        result["actual_rul_seconds"] = windows["rul_seconds"]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return len(result)
