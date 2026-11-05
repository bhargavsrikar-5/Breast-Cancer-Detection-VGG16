"""Dependency-light regression checks for the public research project."""

from __future__ import annotations

import ast
import csv
import json
import re
import unittest
from pathlib import Path
from urllib.parse import unquote, urlsplit

from scripts.generate_result_figures import load_results, partition_results

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_NOTEBOOKS = (
    "00_runtime_smoke_test.ipynb",
    "01_paper_reproduction.ipynb",
    "02_data_augmentation.ipynb",
    "03_model_improvements.ipynb",
    "04_external_validation.ipynb",
)
SOURCE_MODULES = (
    "__init__.py",
    "config.py",
    "preprocessing.py",
    "model.py",
    "train.py",
    "evaluate.py",
    "predict.py",
)
HISTORICAL_METRICS = {
    ("Original VGG16 (default argmax)", "BUSI exploratory test (130 images)"): (
        "0.7615",
        "0.8003",
        "0.7895",
        "0.3571",
        "0.4918",
    ),
    ("Original VGG16 (validation-selected threshold)", "BUSI exploratory test (130 images)"): (
        "0.7385",
        "0.8003",
        "0.5741",
        "0.7381",
        "0.6458",
    ),
    ("Class-weighted VGG16", "BUSI exploratory test (130 images)"): (
        "0.7538",
        "0.8036",
        "0.5962",
        "0.7381",
        "0.6596",
    ),
    ("Class-weighted VGG16 with block-5 fine-tuning", "BUSI exploratory test (130 images)"): (
        "0.8692",
        "0.9351",
        "0.8378",
        "0.7381",
        "0.7848",
    ),
    ("Fine-tuned BUSI model", "BUS-BRA image level (1875 images)"): (
        "0.6400",
        "0.6232",
        "0.4300",
        "0.3443",
        "0.3824",
    ),
    ("Fine-tuned BUSI model", "BUS-BRA case level (1064 cases)"): (
        "0.6880",
        "0.6475",
        "0.5275",
        "0.2807",
        "0.3664",
    ),
    (
        "Letterbox with class weighting and block-5 fine-tuning",
        "BUSI validation (52 images)",
    ): ("0.8269", "0.8689", "", "", ""),
}


def _read_notebook(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssertionError(f"Could not read valid notebook JSON at {path}: {exc}") from exc


def _code_cells(notebook: dict[str, object], path: Path) -> list[str]:
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise AssertionError(f"Notebook has no cell list: {path}")
    sources = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            raise AssertionError(f"Cell {index} is not an object in {path}")
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        try:
            ast.parse(source, filename=f"{path.name}, cell {index}")
        except SyntaxError as exc:
            raise AssertionError(f"Invalid Python in {path.name}, cell {index}: {exc}") from exc
        if cell.get("execution_count") is not None:
            raise AssertionError(f"Public notebook has an executed code cell: {path}, cell {index}")
        if cell.get("outputs"):
            raise AssertionError(f"Public notebook contains saved outputs: {path}, cell {index}")
        sources.append(source)
    return sources


class PublicProjectTests(unittest.TestCase):
    def test_public_entry_points_and_source_modules_exist(self) -> None:
        required_files = (
            "README.md",
            "requirements.txt",
            "requirements-dev.txt",
            ".gitignore",
            "pyproject.toml",
            "CITATION.cff",
            "CONTRIBUTING.md",
            "SECURITY.md",
            ".github/workflows/ci.yml",
            ".github/dependabot.yml",
            "scripts/generate_result_figures.py",
            "docs/architecture.md",
            "results/figures/busi_exploratory_test_comparison.png",
            "results/figures/busbra_external_evaluation.png",
            "results/figures/letterbox_validation_only.png",
            "results/metrics/model_comparison.csv",
            "docs/dataset_documentation.md",
            "docs/experimental_findings.md",
            "docs/licensing.md",
            "docs/paper_methodology.md",
        )
        for relative in required_files:
            with self.subTest(path=relative):
                self.assertTrue(
                    (ROOT / relative).is_file(), f"Required public file is missing: {relative}"
                )
        for name in SOURCE_MODULES:
            path = ROOT / "src" / name
            with self.subTest(path=path):
                self.assertTrue(path.is_file(), f"Required source module is missing: {path}")
                try:
                    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                except SyntaxError as exc:
                    self.fail(f"Source module does not parse ({path}): {exc}")
        for name in PUBLIC_NOTEBOOKS:
            path = ROOT / "notebooks" / name
            with self.subTest(path=path):
                self.assertTrue(path.is_file(), f"Required public notebook is missing: {path}")

    def test_source_and_notebook_checks_require_no_tensorflow_import(self) -> None:
        test_source = (ROOT / "tests" / "test_static_project.py").read_text(encoding="utf-8")
        tree = ast.parse(test_source)
        imported_modules = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported_modules.update(
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        )
        self.assertNotIn(
            "tensorflow",
            imported_modules,
            "Static tests must inspect source, not import TensorFlow-dependent modules.",
        )
        self.assertNotIn(
            "src",
            imported_modules,
            "Static tests should parse source and notebook code without importing project modules.",
        )

    def test_public_notebooks_are_clean_json_with_parseable_code(self) -> None:
        for name in PUBLIC_NOTEBOOKS:
            path = ROOT / "notebooks" / name
            notebook = _read_notebook(path)
            self.assertEqual(notebook.get("nbformat"), 4, f"Expected nbformat 4 in {path}")
            self.assertGreaterEqual(
                notebook.get("nbformat_minor", -1), 4, f"Unsupported notebook format in {path}"
            )
            self.assertGreater(len(notebook.get("cells", [])), 0, f"Notebook is empty: {path}")
            _code_cells(notebook, path)

    def test_smoke_notebook_cannot_start_training(self) -> None:
        path = ROOT / "notebooks" / "00_runtime_smoke_test.ipynb"
        code = "\n".join(_code_cells(_read_notebook(path), path))
        tree = ast.parse(code, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called_name = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else node.func.id
                if isinstance(node.func, ast.Name)
                else ""
            )
            self.assertNotEqual(
                called_name, "fit", f"Training call found in smoke-test notebook: {path}"
            )
            self.assertNotIn(
                called_name,
                {"train_experiment", "fine_tune", "fit_generator"},
                f"Training helper found in smoke-test notebook: {path}",
            )

    def test_readme_local_links_point_to_public_files(self) -> None:
        readme = ROOT / "README.md"
        content = readme.read_text(encoding="utf-8")
        links = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", content)
        self.assertTrue(links, "README should contain navigable Markdown links.")
        for raw_target in links:
            target = raw_target.strip().strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme or not parsed.path:
                continue
            local_path = (readme.parent / unquote(parsed.path)).resolve()
            with self.subTest(link=target):
                self.assertTrue(local_path.exists(), f"README local link is broken: {target}")

    def test_software_citation_credits_only_the_software_author(self) -> None:
        citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        software_authors = re.findall(r'^\s*- name:\s*"?([^"\n]+)', citation, re.MULTILINE)
        self.assertEqual(
            software_authors,
            ["Chegu Bhargav Srikar"],
            "CITATION.cff should credit only the software developer/author/maintainer.",
        )
        paper_notes = (ROOT / "docs" / "paper_methodology.md").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for name in ("Kummari Vishnu", "Chegu Bhargav Srikar", "Pagalla Ashok"):
            self.assertIn(name, paper_notes, f"Published research authorship is missing {name}.")
            self.assertIn(name, readme, f"README research authorship is missing {name}.")
        self.assertIn("software developer, author, and maintainer", readme)

    def test_metrics_csv_has_complete_split_metadata_and_preserved_results(self) -> None:
        path = ROOT / "results" / "metrics" / "model_comparison.csv"
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            expected_columns = {
                "experiment",
                "evaluation_split",
                "accuracy",
                "roc_auc",
                "malignant_precision",
                "malignant_recall",
                "malignant_f1",
                "threshold",
                "confusion_matrix",
                "notes",
            }
            self.assertTrue(
                expected_columns.issubset(reader.fieldnames or []),
                f"Metrics CSV is missing columns: {sorted(expected_columns - set(reader.fieldnames or []))}",
            )
            rows = list(reader)
        self.assertEqual(len(rows), 7, f"Expected 7 historical result rows, found {len(rows)}")
        for row in rows:
            self.assertTrue(
                row["evaluation_split"].strip(), f"Missing split for {row['experiment']}"
            )
            self.assertTrue(row["notes"].strip(), f"Missing result caveat for {row['experiment']}")
        for (name, split), expected in HISTORICAL_METRICS.items():
            with self.subTest(experiment=name, split=split):
                matches = [
                    row
                    for row in rows
                    if row["experiment"] == name and row["evaluation_split"] == split
                ]
                self.assertEqual(
                    len(matches), 1, f"Expected exactly one historical row for {name} / {split}"
                )
                row = matches[0]
                actual = tuple(
                    row[field]
                    for field in (
                        "accuracy",
                        "roc_auc",
                        "malignant_precision",
                        "malignant_recall",
                        "malignant_f1",
                    )
                )
                self.assertEqual(actual, expected, f"Historical metric values changed for {name}")
        self.assertTrue(
            any("BUSI validation" in row["evaluation_split"] for row in rows),
            "The validation-only letterbox result must retain its split label.",
        )
        self.assertTrue(
            any("BUS-BRA image level" in row["evaluation_split"] for row in rows)
            and any("BUS-BRA case level" in row["evaluation_split"] for row in rows),
            "BUS-BRA image- and case-level evaluations must remain distinct.",
        )

    def test_figure_inputs_are_partitioned_by_evaluation_split(self) -> None:
        rows = load_results(ROOT / "results" / "metrics" / "model_comparison.csv")
        busi_test, busbra, letterbox = partition_results(rows)
        self.assertEqual(len(busi_test), 4, "Expected four BUSI exploratory test comparisons.")
        self.assertTrue(
            all("BUSI exploratory test" in row["evaluation_split"] for row in busi_test)
        )
        self.assertEqual(len(busbra), 2, "Expected separate BUS-BRA image- and case-level results.")
        self.assertTrue(all(row["evaluation_split"].startswith("BUS-BRA ") for row in busbra))
        self.assertEqual(len(letterbox), 1, "Expected one letterbox validation-only metric row.")
        self.assertIn("validation", letterbox[0]["evaluation_split"].lower())

    def test_gitignore_protects_local_data_and_reference_material(self) -> None:
        ignore_path = ROOT / ".gitignore"
        rules = set(ignore_path.read_text(encoding="utf-8").splitlines())
        protected = {
            "/data/",
            "/datasets/",
            "/dataset/",
            "/BUSI/",
            "/BUSBRA/",
            "/local_images/",
            "/BreastUltrasound/",
            "**/BUSI/",
            "**/BUSBRA/",
            "*.zip",
            "*.npz",
            "*.tfrecord",
            "*.dcm",
            "*.dicom",
            "*.nii.gz",
            "*.keras",
            "*.h5",
            "*.hdf5",
            "*.ckpt",
            "*.weights",
            "/results/runs/",
            "/results/checkpoints/",
            "**/checkpoints/",
            "/runs/",
            "/outputs/",
            "/Breast_Cancer_VGG16_Colab.ipynb",
            "/notebooks/archive/",
            "/BreastCancerDetection.pdf",
            "/resume-().pdf",
            ".env",
            "credentials.json",
            "token.json",
            "secrets.json",
            "*.pem",
            "*.key",
            "*.tmp",
            "*.bak",
            ".coverage",
            "htmlcov/",
            ".tox/",
            "build/",
            "dist/",
            ".DS_Store",
            "Thumbs.db",
        }
        self.assertTrue(
            protected.issubset(rules),
            f".gitignore is missing protective rules: {sorted(protected - rules)}",
        )
        self.assertNotIn(
            "*.png", rules, "Approved generated project figures must remain publishable."
        )
        self.assertNotIn(
            "*.jpg", rules, "Do not blanket-ignore project assets or hide policy gaps."
        )

    def test_external_threshold_requires_model_matched_busi_validation_provenance(self) -> None:
        source = (ROOT / "src" / "evaluate.py").read_text(encoding="utf-8")
        self.assertIn("def resolve_busbra_threshold", source)
        self.assertIn("BUSI validation", source)
        self.assertIn("No historical default is used.", source)
        self.assertNotIn(
            "0.6782", source, "A historical threshold must not silently become a fallback."
        )
        notebook_path = ROOT / "notebooks" / "04_external_validation.ipynb"
        notebook_code = "\n".join(_code_cells(_read_notebook(notebook_path), notebook_path))
        self.assertIn("metrics.json", notebook_code)
        self.assertIn("BUSI_VALIDATION_THRESHOLD", notebook_code)
        self.assertNotIn('"0.6782"', notebook_code)


if __name__ == "__main__":
    unittest.main()
