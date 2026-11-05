"""Dependency-light tests for single-image prediction helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from src.predict import (
    classify_probabilities,
    predict_image,
    resolve_threshold,
    validate_probabilities,
    validate_threshold,
    write_prediction_json,
)


class PredictionTests(unittest.TestCase):
    def test_probability_validation_accepts_two_class_softmax(self) -> None:
        self.assertEqual(validate_probabilities([[0.82, 0.18]]), (0.82, 0.18))

    def test_probability_validation_rejects_invalid_outputs(self) -> None:
        for output in ([[0.2]], [[0.2, 0.2]], [[-0.1, 1.1]], [[float("nan"), 0.0]]):
            with self.subTest(output=output), self.assertRaises(ValueError):
                validate_probabilities(output)

    def test_threshold_validation(self) -> None:
        self.assertEqual(validate_threshold(0.6), 0.6)
        for value in (-0.01, 1.01, float("inf"), float("nan"), "invalid"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_threshold(value)

    def test_argmax_classification(self) -> None:
        self.assertEqual(classify_probabilities(0.82, 0.18), ("benign", "Argmax"))
        self.assertEqual(classify_probabilities(0.5, 0.5)[0], "benign")

    def test_threshold_classification(self) -> None:
        self.assertEqual(classify_probabilities(0.4, 0.6, 0.55)[0], "malignant")
        self.assertEqual(classify_probabilities(0.4, 0.6, 0.65)[0], "benign")

    def test_missing_checkpoint_and_image_are_reported_without_tensorflow(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            with self.assertRaisesRegex(FileNotFoundError, "checkpoint"):
                predict_image(folder / "missing.keras", folder / "missing.png")
            checkpoint = folder / "model.keras"
            checkpoint.touch()
            with self.assertRaisesRegex(FileNotFoundError, "Image file"):
                predict_image(checkpoint, folder / "missing.png")

    def test_prediction_with_injected_dependencies_and_json_structure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            checkpoint = folder / "model.keras"
            image_file = folder / "image.png"
            checkpoint.touch()
            image_file.touch()
            grayscale = np.full((10, 10), 128, dtype=np.uint8)

            def preprocess(image: np.ndarray, height: int, width: int, mode: str) -> np.ndarray:
                self.assertEqual((height, width, mode), (224, 224, "resize"))
                channel = np.full((height, width), float(image.mean() / 255), dtype=np.float32)
                return np.stack([channel, channel, channel], axis=-1)

            model = Mock()
            model.predict.return_value = np.asarray([[0.82, 0.18]], dtype=np.float32)
            runtime = SimpleNamespace(device=lambda _device: _NullContext())
            loader = Mock(return_value=(model, runtime))
            result = predict_image(
                checkpoint,
                image_file,
                model_loader=loader,
                image_reader=lambda _path: grayscale,
                preprocessor=preprocess,
            )
            self.assertEqual(result["predicted_class"], "benign")
            self.assertEqual(result["decision_method"], "Argmax")
            json_path = write_prediction_json(result, folder / "output" / "prediction.json")
            saved = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertAlmostEqual(saved["probabilities"]["benign"], 0.82, places=6)
            self.assertAlmostEqual(saved["probabilities"]["malignant"], 0.18, places=6)
            self.assertIn("threshold_source", saved)
            self.assertAlmostEqual(saved["confidence"], 0.82, places=6)

    def test_invalid_model_prediction_output_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            checkpoint = folder / "model.keras"
            image_file = folder / "image.png"
            checkpoint.touch()
            image_file.touch()
            model = Mock()
            model.predict.return_value = np.asarray([[0.2, 0.2]], dtype=np.float32)
            runtime = SimpleNamespace(device=lambda _device: _NullContext())
            with self.assertRaisesRegex(ValueError, "sum to 1"):
                predict_image(
                    checkpoint,
                    image_file,
                    model_loader=Mock(return_value=(model, runtime)),
                    image_reader=lambda _path: np.ones((2, 2), dtype=np.uint8),
                    preprocessor=lambda image, height, width, mode: np.zeros(
                        (height, width, 3), dtype=np.float32
                    ),
                )

    def test_metrics_threshold_requires_matching_busi_validation_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            checkpoint = folder / "best_model.keras"
            checkpoint.touch()
            metrics = {
                "validation_metrics": {
                    "selected_f1_threshold": 0.42,
                    "threshold_selection": "Maximum F1 on BUSI validation data",
                    "evaluation_split": "BUSI validation; checkpoint selection",
                },
                "threshold_for_external_evaluation": 0.42,
            }
            metrics_path = folder / "metrics.json"
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
            self.assertEqual(resolve_threshold(checkpoint, metrics_json=metrics_path)[0], 0.42)
            metrics["validation_metrics"]["evaluation_split"] = "BUSI exploratory test"
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "BUSI validation"):
                resolve_threshold(checkpoint, metrics_json=metrics_path)


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: object) -> None:
        return None


if __name__ == "__main__":
    unittest.main()
