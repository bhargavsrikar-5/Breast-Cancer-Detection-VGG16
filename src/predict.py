"""Single-image inference for locally saved VGG16 research checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import numpy as np

CLASS_NAMES = ("benign", "malignant")
IMAGE_SIZE = (224, 224)


def validate_threshold(threshold: float) -> float:
    """Return a finite probability threshold in the inclusive range [0, 1]."""

    try:
        value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("Threshold must be a number between 0 and 1") from exc
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    return value


def validate_probabilities(output: Any) -> tuple[float, float]:
    """Validate one benign/malignant softmax output and return its probabilities."""

    try:
        if hasattr(output, "tolist"):
            output = output.tolist()
    except (TypeError, ValueError) as exc:
        raise ValueError("Model output must contain two numeric class probabilities") from exc
    if not isinstance(output, (list, tuple)) or len(output) != 1:
        shape = getattr(output, "shape", None)
        raise ValueError(f"Expected model prediction shape (1, 2), received {shape or 'invalid'}")
    row = output[0]
    if not isinstance(row, (list, tuple)) or len(row) != 2:
        raise ValueError("Expected model prediction shape (1, 2)")
    try:
        benign, malignant = (float(probability) for probability in row)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Model output must contain two numeric class probabilities") from exc
    if not math.isfinite(benign) or not math.isfinite(malignant):
        raise ValueError("Model output contains NaN or infinity")
    if not 0 <= benign <= 1 or not 0 <= malignant <= 1:
        raise ValueError("Model probabilities must be between 0 and 1")
    if not math.isclose(benign + malignant, 1.0, rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError("Model class probabilities must sum to 1")
    return benign, malignant


def classify_probabilities(
    benign_probability: float,
    malignant_probability: float,
    threshold: float | None = None,
) -> tuple[str, str]:
    """Choose a class by argmax or an explicitly supplied malignant threshold."""

    probabilities = validate_probabilities([[benign_probability, malignant_probability]])
    if threshold is None:
        return CLASS_NAMES[int(probabilities[1] > probabilities[0])], "Argmax"
    value = validate_threshold(threshold)
    predicted = "malignant" if probabilities[1] >= value else "benign"
    return predicted, f"Malignant probability threshold ({value:g})"


def resolve_threshold(
    model_path: str | Path,
    threshold: float | None = None,
    metrics_json: str | Path | None = None,
) -> tuple[float | None, str]:
    """Resolve an explicit threshold or a checkpoint-matched BUSI validation metric.

    Threshold metrics must be explicitly supplied and stored beside the checkpoint.
    No historical threshold is inferred or used as a default.
    """

    if threshold is not None and metrics_json is not None:
        raise ValueError("Choose either --threshold or --metrics-json, not both")
    if threshold is not None:
        return (
            validate_threshold(threshold),
            "Explicit caller-supplied threshold (BUSI validation provenance not verified)",
        )
    if metrics_json is None:
        return None, "Argmax (no threshold supplied)"

    model_file = Path(model_path).expanduser().resolve()
    metrics_file = Path(metrics_json).expanduser().resolve()
    if metrics_file.parent != model_file.parent:
        raise ValueError(
            "The metrics.json file must be in the checkpoint's run directory "
            "to establish checkpoint/threshold matching."
        )
    if not metrics_file.is_file():
        raise FileNotFoundError(f"Metrics JSON file does not exist: {metrics_file}")
    try:
        with metrics_file.open("r", encoding="utf-8") as stream:
            metrics = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read metrics JSON: {metrics_file}") from exc
    if not isinstance(metrics, dict):
        raise ValueError("Metrics JSON must contain a JSON object")

    validation = metrics.get("validation_metrics")
    if not isinstance(validation, dict) and "selected_f1_threshold" in metrics:
        # The letterbox experiment stores its validation record at the top level.
        validation = metrics
    if not isinstance(validation, dict):
        raise ValueError("Metrics JSON has no BUSI validation metrics record")
    split = str(validation.get("evaluation_split", "")).lower()
    method = str(validation.get("threshold_selection", "")).lower()
    if "busi validation" not in split or "validation" not in method:
        raise ValueError(
            "Metrics JSON does not establish a threshold selected on BUSI validation data"
        )

    selected = metrics.get("threshold_for_external_evaluation")
    if selected is None:
        selected = validation.get("selected_f1_threshold")
    if selected is None:
        raise ValueError("Metrics JSON contains no selected BUSI validation threshold")
    value = validate_threshold(selected)
    validation_value = validation.get("selected_f1_threshold")
    if validation_value is not None and not math.isclose(
        value, validate_threshold(validation_value), rel_tol=1e-5, abs_tol=1e-8
    ):
        raise ValueError("Threshold fields disagree in the metrics JSON")
    return value, f"{metrics_file} (BUSI validation metrics)"


def preprocess_image(
    image_path: str | Path,
    *,
    image_reader: Callable[[str], np.ndarray | None] | None = None,
    preprocessor: Callable[..., np.ndarray] | None = None,
) -> np.ndarray:
    """Read and preprocess one grayscale image using the research pipeline helper."""

    path = Path(image_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {path}")
    if image_reader is None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is required for image inference; install requirements.txt"
            ) from exc

        def read_grayscale(filename: str) -> np.ndarray | None:
            return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

        image_reader = read_grayscale
    image = image_reader(str(path))
    if image is None:
        raise ValueError(f"OpenCV could not read image: {path}")
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "NumPy is required for image inference; install requirements.txt"
        ) from exc
    if preprocessor is None:
        # Reuse the existing pipeline implementation (same resize and normalization).
        try:
            from src.preprocessing import preprocess_grayscale_image
        except ImportError as exc:
            raise RuntimeError(
                "TensorFlow and the project dependencies are required for the shared "
                "preprocessing helper; install requirements.txt"
            ) from exc
        preprocessor = preprocess_grayscale_image
    try:
        batch = np.asarray(preprocessor(image, *IMAGE_SIZE, mode="resize"), dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not preprocess image {path}: {exc}") from exc
    if batch.shape != (*IMAGE_SIZE, 3):
        raise ValueError(f"Preprocessor returned shape {batch.shape}; expected {(*IMAGE_SIZE, 3)}")
    if not np.isfinite(batch).all() or (batch < 0).any() or (batch > 1).any():
        raise ValueError("Preprocessed image must contain finite pixel values in [0, 1]")
    if not np.array_equal(batch[..., 0], batch[..., 1]) or not np.array_equal(
        batch[..., 0], batch[..., 2]
    ):
        raise ValueError("Preprocessed grayscale image must have three identical channels")
    return batch[np.newaxis, ...]


def load_checkpoint(model_path: str | Path, device: str = "cpu") -> tuple[Any, Any]:
    """Load a local Keras checkpoint and TensorFlow lazily, without downloading weights."""

    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {path}")
    if device != "cpu":
        raise ValueError("Only --device cpu is currently supported")
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required to load a checkpoint; install requirements.txt"
        ) from exc
    try:
        with tf.device("/CPU:0"):
            model = tf.keras.models.load_model(str(path), compile=False)
    except Exception as exc:
        raise ValueError(f"Could not load model checkpoint {path}: {exc}") from exc
    return model, tf


def predict_image(
    model_path: str | Path,
    image_path: str | Path,
    *,
    threshold: float | None = None,
    metrics_json: str | Path | None = None,
    device: str = "cpu",
    model_loader: Callable[[Path, str], tuple[Any, Any]] | None = None,
    image_reader: Callable[[str], np.ndarray | None] | None = None,
    preprocessor: Callable[..., np.ndarray] | None = None,
) -> dict[str, Any]:
    """Load a checkpoint and predict one image; this function never trains a model."""

    checkpoint = Path(model_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {checkpoint}")
    image_file = Path(image_path).expanduser().resolve()
    if not image_file.is_file():
        raise FileNotFoundError(f"Image file does not exist: {image_file}")
    threshold_value, threshold_source = resolve_threshold(checkpoint, threshold, metrics_json)
    image_batch = preprocess_image(image_file, image_reader=image_reader, preprocessor=preprocessor)
    loader = model_loader or load_checkpoint
    model, runtime = loader(checkpoint, device)
    try:
        with runtime.device("/CPU:0"):
            raw_output = model.predict(image_batch, verbose=0)
    except Exception as exc:
        raise ValueError(f"Model inference failed: {exc}") from exc
    benign_probability, malignant_probability = validate_probabilities(raw_output)
    predicted_class, decision_method = classify_probabilities(
        benign_probability, malignant_probability, threshold_value
    )
    confidence = benign_probability if predicted_class == "benign" else malignant_probability
    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": {
            "benign": benign_probability,
            "malignant": malignant_probability,
        },
        "decision_method": decision_method,
        "threshold": threshold_value,
        "threshold_source": threshold_source,
        "model_path": str(checkpoint),
        "image_path": str(image_file),
    }


def write_prediction_json(result: dict[str, Any], output_path: str | Path) -> Path:
    """Write prediction metadata as indented UTF-8 JSON."""

    path = Path(output_path).expanduser().resolve()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"Could not write prediction JSON to {path}: {exc}") from exc
    return path


def _format_result(result: dict[str, Any]) -> str:
    probabilities = result["probabilities"]
    return "\n".join(
        (
            f"Predicted class: {result['predicted_class'].capitalize()}",
            f"Benign probability: {probabilities['benign']:.4f}",
            f"Malignant probability: {probabilities['malignant']:.4f}",
            f"Confidence: {result['confidence']:.4f}",
            f"Decision method: {result['decision_method']}",
            f"Threshold source: {result['threshold_source']}",
            "Research inference only; not a medical diagnosis.",
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Predict benign/malignant probabilities for one image using a local checkpoint."
    )
    parser.add_argument("--model", required=True, help="Path to a locally saved Keras checkpoint")
    parser.add_argument("--image", required=True, help="Path to one breast ultrasound image")
    decision = parser.add_mutually_exclusive_group()
    decision.add_argument(
        "--threshold", type=float, help="Explicit malignant probability threshold"
    )
    decision.add_argument(
        "--metrics-json", help="Matching run metrics.json with BUSI validation threshold"
    )
    parser.add_argument("--output-json", help="Optional path for prediction and metadata JSON")
    parser.add_argument("--device", choices=("cpu",), default="cpu", help="Inference device (CPU)")
    args = parser.parse_args(argv)
    try:
        result = predict_image(
            args.model,
            args.image,
            threshold=args.threshold,
            metrics_json=args.metrics_json,
            device=args.device,
        )
        print(_format_result(result))
        if args.output_json:
            print(f"JSON saved to: {write_prediction_json(result, args.output_json)}")
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
