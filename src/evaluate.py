"""Evaluation metrics and BUS-BRA external evaluation without training."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.preprocessing import CLASS_NAMES, preprocess_grayscale_image


def calculate_metrics(
    labels: Sequence[int], probabilities: Sequence[float], threshold: float = 0.5
) -> dict[str, Any]:
    """Calculate binary classification metrics; positive class is malignant."""

    y_true = np.asarray(labels, dtype=np.int32)
    malignant_probability = np.asarray(probabilities, dtype=np.float64)
    if (
        y_true.ndim != 1
        or malignant_probability.ndim != 1
        or len(y_true) != len(malignant_probability)
    ):
        raise ValueError("labels and probabilities must be equally sized 1-D arrays")
    if len(y_true) == 0:
        raise ValueError("Cannot evaluate an empty label set")
    if not np.isfinite(malignant_probability).all():
        raise ValueError("Probabilities contain NaN or infinity")
    if ((malignant_probability < 0) | (malignant_probability > 1)).any():
        raise ValueError("Probabilities must be between 0 and 1")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    predictions = (malignant_probability >= threshold).astype(np.int32)
    result: dict[str, Any] = {
        "images_or_cases": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "malignant_precision": float(precision_score(y_true, predictions, zero_division=0)),
        "malignant_recall": float(recall_score(y_true, predictions, zero_division=0)),
        "malignant_f1": float(f1_score(y_true, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true,
            predictions,
            labels=[0, 1],
            target_names=list(CLASS_NAMES),
            output_dict=True,
            zero_division=0,
        ),
    }
    result["roc_auc"] = (
        float(roc_auc_score(y_true, malignant_probability)) if len(np.unique(y_true)) == 2 else None
    )
    return result


def select_f1_threshold(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    """Choose a malignant-probability threshold using validation labels only."""

    y_true = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(probabilities, dtype=np.float64)
    if y_true.ndim != 1 or scores.ndim != 1 or len(y_true) != len(scores) or not len(y_true):
        raise ValueError("labels and probabilities must be equally sized, non-empty 1-D arrays")
    if len(np.unique(y_true)) != 2:
        raise ValueError("Both classes are required to select an F1 threshold")
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if not len(thresholds):
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
    return float(thresholds[int(np.argmax(f1))])


def _case_id(image_id: str) -> str:
    return re.sub(r"[-_](left|right|l|r|s)$", "", image_id, flags=re.IGNORECASE)


def resolve_busbra_threshold(
    model_path: str | Path,
    threshold: float | None = None,
    metrics_json: str | Path | None = None,
) -> tuple[float, str]:
    """Resolve a model-matched BUSI validation threshold or require explicit input.

    A new training run stores ``metrics.json`` beside ``best_model.keras``.
    Manual thresholds are accepted only when explicitly supplied by the caller;
    their BUSI-validation provenance remains the caller's responsibility.
    Supplying ``threshold`` without ``metrics_json`` is an explicit manual choice.
    """

    model_file = Path(model_path).expanduser().resolve()
    if threshold is not None and metrics_json is None:
        value = float(threshold)
        if not 0 <= value <= 1:
            raise ValueError("Explicit BUSI validation threshold must be between 0 and 1")
        return (
            value,
            "Explicit caller-supplied threshold (must be selected on BUSI validation data)",
        )

    metrics_file = (
        Path(metrics_json).expanduser().resolve()
        if metrics_json is not None
        else model_file.parent / "metrics.json"
    )
    if metrics_file.parent != model_file.parent:
        raise ValueError(
            "The metrics.json file must be in the selected checkpoint's run directory "
            "to establish model/threshold matching."
        )
    if metrics_file.is_file():
        try:
            with metrics_file.open("r", encoding="utf-8") as stream:
                metrics = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read threshold metrics file: {metrics_file}") from exc

        validation = metrics.get("validation_metrics")
        if not isinstance(validation, dict) and "selected_f1_threshold" in metrics:
            # Letterbox runs are validation-only, so their validation record is
            # stored at the top level rather than nested beside test metrics.
            validation = metrics
        if not isinstance(validation, dict):
            raise ValueError(
                f"{metrics_file} has no validation_metrics record; provide the correct BUSI "
                "validation threshold explicitly."
            )
        split_name = str(validation.get("evaluation_split", "")).lower()
        selection_method = str(validation.get("threshold_selection", "")).lower()
        if "busi validation" not in split_name or "validation" not in selection_method:
            raise ValueError(
                f"{metrics_file} does not establish a BUSI validation threshold; "
                "provide the correct threshold selected on BUSI validation data."
            )

        value = metrics.get("threshold_for_external_evaluation")
        if value is None:
            value = validation.get("selected_f1_threshold")
        if value is None:
            raise ValueError(
                f"{metrics_file} contains no selected validation threshold; provide the "
                "correct threshold selected on BUSI validation data."
            )
        value = float(value)
        if not 0 <= value <= 1:
            raise ValueError(f"Threshold in {metrics_file} must be between 0 and 1")

        validation_value = validation.get("selected_f1_threshold")
        if validation_value is not None and not np.isclose(value, float(validation_value)):
            raise ValueError(
                f"Threshold fields disagree in {metrics_file}; provide a verified threshold "
                "from BUSI validation data."
            )
        if threshold is not None and not np.isclose(value, float(threshold)):
            raise ValueError(
                f"Explicit threshold {threshold} does not match the threshold for this "
                f"checkpoint in {metrics_file}."
            )
        return value, f"{metrics_file} (BUSI validation_metrics)"

    if threshold is not None:
        value = float(threshold)
        if not 0 <= value <= 1:
            raise ValueError("Explicit BUSI validation threshold must be between 0 and 1")
        return (
            value,
            "Explicit caller-supplied threshold (must be selected on BUSI validation data)",
        )

    raise FileNotFoundError(
        f"No matching metrics.json found beside checkpoint {model_file}. "
        "Provide --metrics-json from this run, or explicitly provide --threshold "
        "selected using BUSI validation data. No historical default is used."
    )


def evaluate_busbra_archive(
    model_path: str | Path,
    archive_path: str | Path,
    output_dir: str | Path,
    threshold: float | None = None,
    batch_size: int = 16,
    image_height: int = 224,
    image_width: int = 224,
    metrics_json: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate a saved BUSI model on BUS-BRA image and averaged case scores.

    BUS-BRA is read only for evaluation. Its images are never used for training
    or threshold selection by this function.
    """

    model_file = Path(model_path).expanduser().resolve()
    zip_file = Path(archive_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if not model_file.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_file}")
    if not zip_file.is_file():
        raise FileNotFoundError(f"BUS-BRA archive not found: {zip_file}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    resolved_threshold, threshold_source = resolve_busbra_threshold(
        model_file, threshold=threshold, metrics_json=metrics_json
    )

    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite existing evaluation artifacts in {destination}; "
            "choose a new empty output directory."
        )
    model = tf.keras.models.load_model(model_file)
    image_rows: list[dict[str, Any]] = []
    buffers: list[np.ndarray] = []
    row_buffer: list[dict[str, Any]] = []

    def flush() -> None:
        if not buffers:
            return
        predictions = model.predict(np.stack(buffers), verbose=0, batch_size=batch_size)
        for row, probability in zip(row_buffer, predictions[:, 1]):
            row["malignant_probability"] = float(probability)
            image_rows.append(row)
        buffers.clear()
        row_buffer.clear()

    try:
        with zipfile.ZipFile(zip_file) as archive:
            names = set(archive.namelist())
            csv_candidates = sorted(name for name in names if name.lower().endswith(".csv"))
            if not csv_candidates:
                raise FileNotFoundError("BUS-BRA archive contains no CSV metadata file")
            metadata_name = next(
                (name for name in csv_candidates if Path(name).name.lower() == "bus_data.csv"),
                csv_candidates[0],
            )
            metadata = pd.read_csv(BytesIO(archive.read(metadata_name)))
            required = {"ID", "Pathology"}
            if not required.issubset(metadata.columns):
                raise ValueError(
                    f"Metadata {metadata_name} must contain columns {sorted(required)}"
                )
            metadata["normalized_pathology"] = (
                metadata["Pathology"].astype(str).str.strip().str.lower()
            )
            metadata = metadata[metadata["normalized_pathology"].isin(CLASS_NAMES)]
            if metadata.empty:
                raise ValueError("BUS-BRA metadata has no benign/malignant labeled images")

            for row in metadata.itertuples(index=False):
                image_id = str(getattr(row, "ID")).strip()
                pathology = str(getattr(row, "normalized_pathology"))
                image_name = f"BUSBRA/Images/{image_id}.png"
                if image_name not in names:
                    raise FileNotFoundError(f"Image listed in metadata is missing: {image_name}")
                encoded = np.frombuffer(archive.read(image_name), dtype=np.uint8)
                grayscale = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
                if grayscale is None:
                    raise ValueError(f"OpenCV could not decode image from archive: {image_name}")
                image = preprocess_grayscale_image(
                    grayscale, image_height, image_width, mode="resize"
                )
                buffers.append(image)
                row_buffer.append(
                    {
                        "image_id": image_id,
                        "case_id": _case_id(image_id),
                        "pathology": pathology,
                        "label": int(pathology == "malignant"),
                    }
                )
                if len(buffers) >= batch_size:
                    flush()
            flush()
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Not a valid ZIP archive: {zip_file}") from exc

    image_df = pd.DataFrame(image_rows)
    image_metrics = calculate_metrics(
        image_df["label"].to_numpy(),
        image_df["malignant_probability"].to_numpy(),
        resolved_threshold,
    )
    if (image_df.groupby("case_id")["label"].nunique() > 1).any():
        raise ValueError("Conflicting pathology labels were found within BUS-BRA cases")
    case_df = image_df.groupby("case_id", as_index=False).agg(
        label=("label", "first"),
        malignant_probability=("malignant_probability", "mean"),
        images=("image_id", "count"),
    )
    case_metrics = calculate_metrics(
        case_df["label"].to_numpy(),
        case_df["malignant_probability"].to_numpy(),
        resolved_threshold,
    )

    image_df.to_csv(destination / "image_level_predictions.csv", index=False)
    case_df.to_csv(destination / "case_level_predictions.csv", index=False)
    report = {
        "model": str(model_file),
        "dataset": "BUS-BRA external evaluation",
        "threshold": float(resolved_threshold),
        "threshold_source": threshold_source,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "image_level": image_metrics,
        "case_level": case_metrics,
        "note": "No BUS-BRA data were used for training or threshold selection.",
    }
    with (destination / "external_evaluation.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Saved Keras .keras checkpoint")
    parser.add_argument("--busbra-zip", required=True, help="Path to the BUS-BRA ZIP archive")
    parser.add_argument("--output-dir", required=True, help="External evaluation output folder")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Explicit threshold selected on BUSI validation data; never tune on BUS-BRA",
    )
    parser.add_argument(
        "--metrics-json",
        default=None,
        help="Matching metrics.json beside the selected checkpoint (preferred threshold source)",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    report = evaluate_busbra_archive(
        model_path=args.model,
        archive_path=args.busbra_zip,
        output_dir=args.output_dir,
        threshold=args.threshold,
        metrics_json=args.metrics_json,
        batch_size=args.batch_size,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
