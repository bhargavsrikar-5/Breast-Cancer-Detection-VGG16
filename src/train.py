"""Train explicitly requested BUSI experiments; importing this module trains nothing."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

from src.config import ProjectConfig
from src.evaluate import calculate_metrics, select_f1_threshold
from src.model import build_vgg16_model
from src.preprocessing import (
    CLASS_NAMES,
    DataSplits,
    ImageRecord,
    discover_busi_images,
    make_stratified_splits,
    make_tf_dataset,
)


EXPERIMENTS = ("baseline", "augmentation", "class_weighted", "finetuned", "letterbox")


def _unique_run_dir(base: Path, experiment: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = base / experiment / stamp
    suffix = 1
    while path.exists():
        path = base / experiment / f"{stamp}_{suffix}"
        suffix += 1
    path.mkdir(parents=True, exist_ok=False)
    return path


def _weights(records: tuple[ImageRecord, ...]) -> dict[int, float]:
    labels = np.asarray([record.label for record in records], dtype=np.int32)
    classes = np.unique(labels)
    values = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return {int(class_id): float(weight) for class_id, weight in zip(classes, values)}


def _callbacks(run_dir: Path) -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]


def _fit_stage(
    model: tf.keras.Model,
    train_data: tf.data.Dataset,
    validation_data: tf.data.Dataset,
    run_dir: Path,
    epochs: int,
    class_weights: dict[int, float] | None = None,
) -> dict[str, list[float]]:
    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=epochs,
        callbacks=_callbacks(run_dir),
        class_weight=class_weights,
        verbose=2,
    )
    return {key: [float(value) for value in values] for key, values in history.history.items()}


def _save_curves(histories: dict[str, dict[str, list[float]]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for stage, history in histories.items():
        for metric, axis in (("loss", axes[0]), ("accuracy", axes[1])):
            if metric in history:
                axis.plot(history[metric], label=f"{stage} train")
            validation_metric = f"val_{metric}"
            if validation_metric in history:
                axis.plot(history[validation_metric], linestyle="--", label=f"{stage} validation")
    axes[0].set_title("Training and validation loss")
    axes[1].set_title("Training and validation accuracy")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _evaluate_test(
    model: tf.keras.Model, test_data: tf.data.Dataset, test_records: tuple[ImageRecord, ...]
) -> dict[str, Any]:
    true_labels = np.asarray([record.label for record in test_records], dtype=np.int32)
    probabilities = model.predict(test_data, verbose=0)[:, 1]
    # The standard class decision is argmax, equivalent to threshold 0.5 here.
    result = calculate_metrics(true_labels, probabilities, threshold=0.5)
    result["roc_auc"] = float(roc_auc_score(true_labels, probabilities))
    result["class_order"] = list(CLASS_NAMES)
    result["evaluation_split"] = "BUSI exploratory test split; reused across notebook experiments"
    return result


def _validation_metrics(
    model: tf.keras.Model,
    validation_data: tf.data.Dataset,
    validation_records: tuple[ImageRecord, ...],
) -> tuple[dict[str, Any], float]:
    labels = np.asarray([record.label for record in validation_records], dtype=np.int32)
    probabilities = model.predict(validation_data, verbose=0)[:, 1]
    threshold = select_f1_threshold(labels, probabilities)
    result = calculate_metrics(labels, probabilities, threshold=0.5)
    result["roc_auc"] = float(roc_auc_score(labels, probabilities))
    result["selected_f1_threshold"] = threshold
    result["threshold_selection"] = "Maximum F1 on BUSI validation data"
    result["evaluation_split"] = "BUSI validation; used for early stopping and checkpoint selection"
    return result, threshold


def train_experiment(config: ProjectConfig, experiment: str) -> Path:
    """Train one explicitly selected experiment into a new timestamped folder."""

    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment {experiment!r}; choose from {EXPERIMENTS}")
    tf.keras.utils.set_random_seed(config.seed)
    records = discover_busi_images(config.data_root)
    splits: DataSplits = make_stratified_splits(
        records,
        seed=config.seed,
        test_fraction=config.test_fraction,
        validation_fraction_of_train=config.validation_fraction,
    )
    preprocess_mode = "letterbox" if experiment == "letterbox" else "resize"
    augmentation = experiment == "augmentation"
    train_data = make_tf_dataset(
        splits.train, config.batch_size, config.image_height, config.image_width,
        training=True, seed=config.seed, mode=preprocess_mode,
    )
    validation_data = make_tf_dataset(
        splits.validation, config.batch_size, config.image_height, config.image_width,
        mode=preprocess_mode,
    )
    test_data = make_tf_dataset(
        splits.test, config.batch_size, config.image_height, config.image_width,
        mode=preprocess_mode,
    )

    run_dir = _unique_run_dir(config.output_root, experiment)
    class_weights = _weights(splits.train) if experiment in ("class_weighted", "finetuned", "letterbox") else None
    histories: dict[str, dict[str, list[float]]] = {}

    # The paper-faithful baseline is frozen and has no class weighting,
    # fine-tuning, letterbox transform, or augmentation.
    model = build_vgg16_model(
        learning_rate=1e-4,
        fine_tune_block5=False,
        augmentation=augmentation,
        input_shape=(config.image_height, config.image_width, 3),
    )
    stage1_epochs = 20
    histories["frozen_backbone"] = _fit_stage(
        model, train_data, validation_data, run_dir, stage1_epochs, class_weights
    )

    # Fine-tuning and the historical letterbox workflow use a separate
    # low-learning-rate block-5 stage, not the paper-faithful baseline.
    if experiment in ("finetuned", "letterbox"):
        model = tf.keras.models.load_model(run_dir / "best_model.keras")
        backbone = next(layer for layer in model.layers if isinstance(layer, tf.keras.Model) and layer.name == "vgg16")
        backbone.trainable = True
        for layer in backbone.layers:
            layer.trainable = layer.name.startswith("block5_")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        histories["block5_fine_tuning"] = _fit_stage(
            model, train_data, validation_data, run_dir, epochs=10, class_weights=class_weights
        )

    best_model = tf.keras.models.load_model(run_dir / "best_model.keras")
    validation_metrics, selected_threshold = _validation_metrics(
        best_model, validation_data, splits.validation
    )
    if experiment == "letterbox":
        # Preserve the historical validation-only status; do not consume the
        # reused BUSI test set for this preprocessing experiment.
        metrics = dict(validation_metrics)
        metrics["note"] = (
            "Letterbox results are validation-only; this split was also used for "
            "early stopping, checkpoint selection, and threshold selection."
        )
    else:
        metrics = _evaluate_test(best_model, test_data, splits.test)
        metrics["validation_metrics"] = validation_metrics
        metrics["validation_selected_threshold"] = selected_threshold
        metrics["threshold_for_external_evaluation"] = selected_threshold
    metrics.update(
        {
            "experiment": experiment,
            "seed": config.seed,
            "train_images": len(splits.train),
            "validation_images": len(splits.validation),
            "test_images": len(splits.test) if experiment != "letterbox" else None,
            "class_weights": class_weights,
            "preprocessing": preprocess_mode,
            "augmentation": augmentation,
            "fine_tuned_block5": experiment in ("finetuned", "letterbox"),
        }
    )
    if experiment == "letterbox":
        metrics["independent_test_evaluation"] = False
        metrics["note"] = (
            "No independent test evaluation was performed. Validation data were used for "
            "early stopping and checkpoint/model selection, and also for threshold selection. "
            "Validation performance must not be described as independent generalization performance."
        )
    else:
        metrics["independent_test_evaluation"] = False
        metrics["note"] = (
            "BUSI test results are exploratory because the same test split was reused across "
            "historical experiments; do not describe them as an untouched final evaluation."
        )
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
    with (run_dir / "training_history.json").open("w", encoding="utf-8") as stream:
        json.dump(histories, stream, indent=2)
    with (run_dir / "experiment_config.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "experiment": experiment,
                "seed": config.seed,
                "data_root": str(config.data_root),
                "split": {
                    "test_fraction": config.test_fraction,
                    "validation_fraction_of_training": config.validation_fraction,
                    "stratified": True,
                    "grouped_by_case": False,
                },
                "image_size": [config.image_height, config.image_width],
                "batch_size": config.batch_size,
                "class_order": list(CLASS_NAMES),
            },
            stream,
            indent=2,
        )
    _save_curves(histories, run_dir / "training_curves.png")
    print(f"Saved this run to: {run_dir}")
    print(json.dumps(metrics, indent=2))
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="baseline")
    parser.add_argument("--data-root", default=None, help="BUSI root; alternatively set BUSI_ROOT")
    parser.add_argument("--output-root", default=None, help="Run output root")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    config = ProjectConfig.from_env(args.data_root, args.output_root, args.seed)
    print(f"Experiment: {args.experiment}")
    print(f"BUSI root: {config.data_root}")
    print(f"Output root: {config.output_root}")
    train_experiment(config, args.experiment)


if __name__ == "__main__":
    main()

