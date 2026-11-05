"""BUSI discovery, splitting, and batch-based OpenCV/TensorFlow input pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


CLASS_NAMES = ("benign", "malignant")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PreprocessMode = Literal["resize", "letterbox"]


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: int


@dataclass(frozen=True)
class DataSplits:
    train: tuple[ImageRecord, ...]
    validation: tuple[ImageRecord, ...]
    test: tuple[ImageRecord, ...]


def _find_class_dir(root: Path, aliases: Sequence[str]) -> Path | None:
    candidates = sorted(
        (p for p in root.rglob("*") if p.is_dir() and p.name.lower() in aliases),
        key=lambda p: (len(p.parts), str(p).lower()),
    )
    return candidates[0] if candidates else None


def discover_busi_images(data_root: str | Path) -> list[ImageRecord]:
    """Find benign and malignant BUSI images, excluding masks and normal scans."""

    root = Path(data_root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"BUSI data directory does not exist: {root}")

    class_dirs = {
        "benign": _find_class_dir(root, ("benign", "benign_tumor")),
        "malignant": _find_class_dir(root, ("malignant", "malignant_tumor")),
    }
    missing = [name for name, folder in class_dirs.items() if folder is None]
    if missing:
        raise FileNotFoundError(
            f"Could not find BUSI class folder(s) {missing} under {root}. "
            "Expected benign/benign_tumor and malignant/malignant_tumor folders."
        )

    records: list[ImageRecord] = []
    for class_name in CLASS_NAMES:
        folder = class_dirs[class_name]
        assert folder is not None
        paths = sorted(
            path
            for path in folder.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_SUFFIXES
            and "_mask" not in path.stem.lower()
        )
        records.extend(ImageRecord(path, CLASS_TO_INDEX[class_name]) for path in paths)

    counts = {name: sum(r.label == index for r in records) for index, name in enumerate(CLASS_NAMES)}
    if any(counts[name] == 0 for name in CLASS_NAMES):
        raise ValueError(f"Both binary classes must have images; found counts: {counts}")
    return records


def make_stratified_splits(
    records: Sequence[ImageRecord],
    seed: int = 42,
    test_fraction: float = 0.20,
    validation_fraction_of_train: float = 0.10,
) -> DataSplits:
    """Split by image with stratification; validation is carved from train.

    This intentionally mirrors the current research workflow. It is not a
    patient/lesion-grouped split, so correlated BUSI views could cross splits.
    """

    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if not 0 < validation_fraction_of_train < 1:
        raise ValueError("validation_fraction_of_train must be between 0 and 1")
    if len(records) < 6:
        raise ValueError("At least six labeled images are required for stratified splitting")

    labels = np.asarray([record.label for record in records], dtype=np.int32)
    indexes = np.arange(len(records))
    train_val_idx, test_idx = train_test_split(
        indexes,
        test_size=test_fraction,
        random_state=seed,
        stratify=labels,
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=validation_fraction_of_train,
        random_state=seed,
        stratify=labels[train_val_idx],
    )
    return DataSplits(
        train=tuple(records[int(i)] for i in train_idx),
        validation=tuple(records[int(i)] for i in val_idx),
        test=tuple(records[int(i)] for i in test_idx),
    )


def _decode_and_preprocess(
    path_bytes: bytes,
    image_height: int,
    image_width: int,
    mode: PreprocessMode,
) -> np.ndarray:
    path = Path(path_bytes.decode("utf-8"))
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"OpenCV could not read image: {path}")
    return preprocess_grayscale_image(image, image_height, image_width, mode)


def preprocess_grayscale_image(
    image: np.ndarray,
    image_height: int = 224,
    image_width: int = 224,
    mode: PreprocessMode = "resize",
) -> np.ndarray:
    """Convert a grayscale image to three repeated channels and scale to [0, 1]."""

    if image.ndim != 2:
        raise ValueError(f"Expected a grayscale 2-D image, received shape {image.shape}")
    if image.size == 0:
        raise ValueError("Image is empty")
    if image_height <= 0 or image_width <= 0:
        raise ValueError("Image dimensions must be positive")

    if mode == "resize":
        resized = cv2.resize(image, (image_width, image_height), interpolation=cv2.INTER_AREA)
    elif mode == "letterbox":
        scale = min(image_width / image.shape[1], image_height / image.shape[0])
        new_width = max(1, round(image.shape[1] * scale))
        new_height = max(1, round(image.shape[0] * scale))
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        resized = np.zeros((image_height, image_width), dtype=np.uint8)
        top = (image_height - new_height) // 2
        left = (image_width - new_width) // 2
        resized[top : top + new_height, left : left + new_width] = resized_image
    else:
        raise ValueError(f"Unknown preprocessing mode: {mode}")

    three_channel = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    return three_channel.astype(np.float32) / 255.0


def make_tf_dataset(
    records: Sequence[ImageRecord],
    batch_size: int = 16,
    image_height: int = 224,
    image_width: int = 224,
    training: bool = False,
    seed: int = 42,
    mode: PreprocessMode = "resize",
) -> tf.data.Dataset:
    """Create a streaming, parallel OpenCV loader with batching and prefetch."""

    if not records:
        raise ValueError("Cannot build a TensorFlow dataset from zero image records")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    paths = np.asarray([str(record.path) for record in records], dtype=np.str_)
    labels = np.asarray([record.label for record in records], dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(records), seed=seed, reshuffle_each_iteration=True)

    def load_one(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.numpy_function(
            func=lambda raw_path: _decode_and_preprocess(
                raw_path, image_height, image_width, mode
            ),
            inp=[path],
            Tout=tf.float32,
        )
        image.set_shape((image_height, image_width, 3))
        label_one_hot = tf.one_hot(label, depth=len(CLASS_NAMES), dtype=tf.float32)
        return image, label_one_hot

    dataset = dataset.map(load_one, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    options = tf.data.Options()
    options.experimental_deterministic = True
    return dataset.with_options(options).prefetch(tf.data.AUTOTUNE)

