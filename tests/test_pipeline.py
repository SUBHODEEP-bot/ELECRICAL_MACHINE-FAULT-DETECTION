import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fault_detection.pipeline import (
    combine_csv,
    make_windows,
    predict,
    prepare_csv,
    train_model,
)


class PipelineTests(unittest.TestCase):
    def test_prepare_maps_labels_and_derives_rul(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw" / "normal"
            source.mkdir(parents=True)
            pd.DataFrame({"signal": [1.0, 2.0, 3.0, 4.0]}).to_csv(
                source / "asset_a.csv", index=False
            )
            target = root / "prepared.csv"
            count = prepare_csv(
                root / "raw",
                target,
                dataset="test",
                sensor_columns=["signal"],
                derive_rul=True,
                sample_rate=2,
            )
            prepared = pd.read_csv(target)
            self.assertEqual(count, 4)
            self.assertEqual(set(prepared["fault_type"]), {"normal"})
            self.assertEqual(prepared["rul_seconds"].tolist(), [1.5, 1.0, 0.5, 0.0])

    def test_window_extraction_has_features_and_labels(self):
        samples = 32
        data = pd.DataFrame(
            {
                "dataset": "test",
                "asset_id": "a",
                "run_id": "r",
                "timestamp": np.arange(samples),
                "fault_type": "normal",
                "rul_seconds": np.arange(samples, 0, -1),
                "vibration": np.sin(np.arange(samples) * 2 * np.pi / 8),
            }
        )
        windows, features = make_windows(data, 16, 8, 100)
        self.assertEqual(len(windows), 3)
        self.assertTrue(features)
        self.assertEqual(windows["fault_type"].unique().tolist(), ["normal"])

    def test_combine_rejects_mismatched_sensor_schemas(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = {
                "dataset": ["test"],
                "asset_id": ["asset"],
                "run_id": ["run"],
                "timestamp": [0],
                "fault_type": ["normal"],
                "rul_seconds": [np.nan],
            }
            first = root / "first.csv"
            second = root / "second.csv"
            pd.DataFrame({**canonical, "vibration": [1.0]}).to_csv(first, index=False)
            pd.DataFrame({**canonical, "current": [2.0]}).to_csv(second, index=False)
            with self.assertRaisesRegex(ValueError, "Sensor schema mismatch"):
                combine_csv([first, second], root / "combined.csv")

    def test_train_and_predict_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            rng = np.random.default_rng(42)
            for asset_index in range(4):
                label = "normal" if asset_index < 2 else "bearing_fault"
                for index in range(64):
                    base = 0.1 if label == "normal" else 3.0
                    rows.append(
                        {
                            "dataset": "test",
                            "asset_id": f"asset_{asset_index}",
                            "run_id": "run_0",
                            "timestamp": index,
                            "fault_type": label,
                            "rul_seconds": float(64 - index) if label != "normal" else np.nan,
                            "vibration": base + rng.normal(0, 0.1),
                        }
                    )
            input_path = root / "canonical.csv"
            pd.DataFrame(rows).to_csv(input_path, index=False)
            model_dir = root / "model"
            metrics = train_model(
                input_path,
                model_dir,
                window_size=8,
                stride=4,
                sample_rate=100,
            )
            self.assertTrue((model_dir / "model.joblib").is_file())
            self.assertIn("classification", metrics)
            output_path = root / "predictions.csv"
            count = predict(input_path, model_dir, output_path)
            predictions = pd.read_csv(output_path)
            self.assertGreater(count, 0)
            self.assertIn("predicted_fault", predictions)
            self.assertIn("anomaly", predictions)
            self.assertIn("predicted_rul_seconds", predictions)


if __name__ == "__main__":
    unittest.main()
