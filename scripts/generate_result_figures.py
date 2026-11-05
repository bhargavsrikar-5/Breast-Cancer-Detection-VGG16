"""Generate labeled visual summaries from the tracked historical metrics CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "results" / "metrics" / "model_comparison.csv"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "figures"
REQUIRED_COLUMNS = {
    "experiment",
    "evaluation_split",
    "accuracy",
    "roc_auc",
    "malignant_precision",
    "malignant_recall",
    "malignant_f1",
    "notes",
}


def load_results(csv_path: str | Path = DEFAULT_CSV) -> list[dict[str, str]]:
    """Read and validate the tracked metrics table without altering its values."""

    source = Path(csv_path)
    with source.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"Metrics CSV {source} is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Metrics CSV contains no result rows: {source}")
    for line_number, row in enumerate(rows, start=2):
        if not row.get("experiment", "").strip():
            raise ValueError(f"Metrics CSV row {line_number} has no experiment name")
        if not row.get("evaluation_split", "").strip():
            raise ValueError(f"Metrics CSV row {line_number} has no evaluation split")
    return rows


def partition_results(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Select BUSI exploratory test, BUS-BRA external, and letterbox validation rows."""

    busi_test = [
        row for row in rows if row["evaluation_split"].lower().startswith("busi exploratory test")
    ]
    busbra = [row for row in rows if row["evaluation_split"].lower().startswith("bus-bra ")]
    letterbox_validation = [
        row
        for row in rows
        if "letterbox" in row["experiment"].lower()
        and "busi validation" in row["evaluation_split"].lower()
    ]
    if not busi_test:
        raise ValueError("No BUSI exploratory test rows were found in the metrics CSV")
    if not busbra:
        raise ValueError("No BUS-BRA external evaluation rows were found in the metrics CSV")
    if len(letterbox_validation) != 1:
        raise ValueError(
            "Expected exactly one letterbox BUSI validation-only row; "
            f"found {len(letterbox_validation)}"
        )
    return busi_test, busbra, letterbox_validation


def _number(row: dict[str, str], column: str) -> float | None:
    value = row.get(column, "").strip()
    return float(value) if value else None


def generate_figures(
    csv_path: str | Path = DEFAULT_CSV, output_dir: str | Path = DEFAULT_OUTPUT_DIR
) -> list[Path]:
    """Render three figures from existing CSV values; no model or dataset is loaded."""

    import matplotlib.pyplot as plt
    import numpy as np

    source = Path(csv_path).resolve()
    destination = Path(output_dir).resolve()
    rows = load_results(source)
    busi_test, busbra, letterbox_rows = partition_results(rows)
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "#f7f9fc",
            "axes.edgecolor": "#d0d7de",
            "axes.grid": True,
            "grid.color": "#d8dee4",
            "grid.alpha": 0.65,
        }
    )
    generated: list[Path] = []

    # BUSI test experiments share a repeatedly used exploratory split.
    labels = [
        row["experiment"]
        .replace("Class-weighted VGG16 with ", "Weighted + ")
        .replace("Original VGG16", "Original")
        for row in busi_test
    ]
    metrics = (
        ("accuracy", "Accuracy"),
        ("roc_auc", "ROC-AUC"),
        ("malignant_recall", "Malignant recall"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.5), layout="constrained")
    positions = np.arange(len(labels))
    colors = ["#2b6cb0", "#4c9f70", "#d69e2e", "#805ad5"]
    for axis, (column, title) in zip(axes, metrics):
        values = [_number(row, column) for row in busi_test]
        axis.barh(positions, values, color=colors[: len(values)], height=0.64)
        axis.set_yticks(positions)
        if axis is axes[0]:
            axis.set_yticklabels(labels)
        else:
            axis.tick_params(axis="y", labelleft=False)
        axis.invert_yaxis()
        axis.set_xlim(0, 1)
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel("Score (0–1)")
        for y, value in zip(positions, values):
            axis.text(value + 0.02, y, f"{value:.3f}", va="center", fontsize=8)
    figure.suptitle("BUSI exploratory test comparison", fontsize=15, fontweight="bold")
    figure.text(
        0.5,
        -0.02,
        "Historical metrics from results/metrics/model_comparison.csv. The same exploratory test split was reused across experiments; comparisons are not confirmatory.",
        ha="center",
        fontsize=8,
        color="#444444",
    )
    path = destination / "busi_exploratory_test_comparison.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    generated.append(path)

    # BUS-BRA rows are external results and must remain separate from BUSI development.
    external_levels = ("BUS-BRA image level", "BUS-BRA case level")
    external_rows = []
    for level in external_levels:
        match = [row for row in busbra if row["evaluation_split"].startswith(level)]
        if len(match) != 1:
            raise ValueError(f"Expected one CSV record for {level}; found {len(match)}")
        external_rows.append(match[0])
    external_metrics = (
        ("accuracy", "Accuracy"),
        ("roc_auc", "ROC-AUC"),
        ("malignant_recall", "Malignant recall"),
    )
    figure, axis = plt.subplots(figsize=(9, 5.5), layout="constrained")
    width = 0.23
    group_positions = np.arange(len(external_metrics))
    level_colors = ("#2b6cb0", "#dd6b20")
    for level_index, row in enumerate(external_rows):
        values = [_number(row, column) for column, _ in external_metrics]
        bars = axis.bar(
            group_positions + (level_index - 0.5) * width,
            values,
            width,
            label=external_levels[level_index].replace("BUS-BRA ", ""),
            color=level_colors[level_index],
        )
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    axis.set_xticks(group_positions, [title for _, title in external_metrics])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score (0–1)")
    axis.set_title("BUS-BRA external evaluation by aggregation level", fontweight="bold")
    axis.legend(title="Evaluation unit", frameon=False)
    axis.text(
        0.01,
        -0.18,
        "External evaluation of the BUSI-trained model; threshold was selected on BUSI validation data. Image- and case-level results use different aggregation units.",
        transform=axis.transAxes,
        fontsize=8,
        color="#444444",
    )
    path = destination / "busbra_external_evaluation.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    generated.append(path)

    # Explicitly label the only letterbox row as validation-only.
    letterbox = letterbox_rows[0]
    letterbox_metrics = (("accuracy", "Accuracy"), ("roc_auc", "ROC-AUC"))
    values = [_number(letterbox, column) for column, _ in letterbox_metrics]
    figure, axis = plt.subplots(figsize=(8, 4.8), layout="constrained")
    bars = axis.bar(
        [title for _, title in letterbox_metrics], values, color=("#b7791f", "#805ad5"), width=0.55
    )
    axis.bar_label(bars, fmt="%.3f", padding=4, fontsize=10)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score (0–1)")
    axis.set_title(
        "LETTERBOX RESULT — BUSI VALIDATION ONLY", color="#8a4b08", fontweight="bold", pad=14
    )
    axis.text(
        0.5,
        0.93,
        "Not an independent test result",
        transform=axis.transAxes,
        ha="center",
        color="#8a4b08",
        fontweight="bold",
    )
    axis.text(
        0.5,
        -0.20,
        "Source: results/metrics/model_comparison.csv. Validation also informed checkpoint selection and threshold selection; class weighting and block-5 fine-tuning were combined with letterbox preprocessing.",
        transform=axis.transAxes,
        ha="center",
        fontsize=8,
        color="#444444",
        wrap=True,
    )
    path = destination / "letterbox_validation_only.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    generated.append(path)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Historical comparison CSV")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Figure output directory"
    )
    args = parser.parse_args()
    for path in generate_figures(args.csv, args.output_dir):
        print(f"Generated: {path}")


if __name__ == "__main__":
    main()
