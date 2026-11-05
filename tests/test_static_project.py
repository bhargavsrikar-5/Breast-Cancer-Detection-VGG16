"""Dependency-free structural checks; these do not import TensorFlow or train."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ProjectStructureTests(unittest.TestCase):
    def test_required_source_modules_exist_and_parse(self) -> None:
        expected = ("config.py", "preprocessing.py", "model.py", "train.py", "evaluate.py")
        for name in expected:
            path = ROOT / "src" / name
            self.assertTrue(path.is_file(), f"Missing {path}")
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_notebooks_are_valid_json_and_code_parses(self) -> None:
        expected = (
            "00_runtime_smoke_test.ipynb",
            "01_paper_reproduction.ipynb",
            "02_data_augmentation.ipynb",
            "03_model_improvements.ipynb",
            "04_external_validation.ipynb",
        )
        for name in expected:
            path = ROOT / "notebooks" / name
            self.assertTrue(path.is_file(), f"Missing {path}")
            notebook = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(notebook["nbformat"], 4)
            self.assertGreaterEqual(notebook["nbformat_minor"], 4)
            self.assertIsInstance(notebook["cells"], list)
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    self.assertIsNone(cell["execution_count"])
                    self.assertEqual(cell["outputs"], [])
                    ast.parse("".join(cell["source"]), filename=f"{name} cell")

    def test_expected_project_files_exist(self) -> None:
        for relative in (
            "README.md",
            "requirements.txt",
            ".gitignore",
            "docs/paper_methodology.md",
            "docs/experimental_findings.md",
            "docs/dataset_documentation.md",
            "results/metrics/model_comparison.csv",
            "notebooks/archive/Breast_Cancer_VGG16_Colab_original.ipynb",
            "notebooks/00_runtime_smoke_test.ipynb",
        ):
            self.assertTrue((ROOT / relative).is_file(), f"Missing {relative}")

    def test_metrics_csv_has_split_and_experiment_columns(self) -> None:
        path = ROOT / "results/metrics/model_comparison.csv"
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertGreaterEqual(len(rows), 7)
        self.assertTrue(all(row["evaluation_split"] for row in rows))
        self.assertTrue(all(row["experiment"] for row in rows))

    def test_gitignore_protects_local_data_and_reference_files(self) -> None:
        rules = set((ROOT / ".gitignore").read_text(encoding="utf-8").splitlines())
        expected = {
            "/data/", "/datasets/", "/BUSI/", "/BUSBRA/",
            "/local_images/", "/BreastUltrasound/",
            "/Breast_Cancer_VGG16_Colab.ipynb", "/notebooks/archive/",
            "/BreastCancerDetection.pdf", "/resume-().pdf",
        }
        self.assertTrue(expected.issubset(rules))
        self.assertNotIn("*.png", rules)
        self.assertNotIn("*.jpg", rules)

    def test_public_notebooks_have_no_embedded_images_or_threshold_default(self) -> None:
        for path in (ROOT / "notebooks").glob("*.ipynb"):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for cell in notebook["cells"]:
                for output in cell.get("outputs", []):
                    self.assertNotIn("image/png", output.get("data", {}), str(path))
        external = json.loads(
            (ROOT / "notebooks/04_external_validation.ipynb").read_text(encoding="utf-8")
        )
        code = "\n".join(
            "".join(cell["source"])
            for cell in external["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertNotIn('"0.6782"', code)
        self.assertIn("metrics.json", code)

    def test_runtime_smoke_notebook_never_trains(self) -> None:
        path = ROOT / "notebooks/00_runtime_smoke_test.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertNotIn(".fit(", code)
        self.assertNotIn("train_experiment(", code)
        self.assertNotIn("src.train", code)
        self.assertEqual(notebook["cells"][-1]["cell_type"], "code")

    def test_local_original_notebooks_remain_byte_identical(self) -> None:
        original = ROOT / "Breast_Cancer_VGG16_Colab.ipynb"
        archived = ROOT / "notebooks/archive/Breast_Cancer_VGG16_Colab_original.ipynb"
        self.assertEqual(
            hashlib.sha256(original.read_bytes()).digest(),
            hashlib.sha256(archived.read_bytes()).digest(),
        )

    def test_external_threshold_has_no_historical_fallback(self) -> None:
        source = (ROOT / "src/evaluate.py").read_text(encoding="utf-8")
        self.assertIn("resolve_busbra_threshold", source)
        self.assertIn("No historical default is used.", source)
        self.assertNotIn("0.6782", source)


if __name__ == "__main__":
    unittest.main()

