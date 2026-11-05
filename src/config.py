"""Runtime configuration with environment-variable and CLI-friendly defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ProjectConfig:
    """Paths and reproducibility settings for a project run."""

    data_root: Path
    output_root: Path
    seed: int = 42
    image_height: int = 224
    image_width: int = 224
    batch_size: int = 16
    max_epochs: int = 20
    validation_fraction: float = 0.10
    test_fraction: float = 0.20

    @classmethod
    def from_env(
        cls,
        data_root: str | Path | None = None,
        output_root: str | Path | None = None,
        seed: int | None = None,
    ) -> "ProjectConfig":
        """Build configuration from explicit arguments or environment variables.

        ``BUSI_ROOT`` may point either to the BUSI directory or to a parent
        directory containing the benign and malignant class folders.
        ``PROJECT_OUTPUT_ROOT`` selects where run artifacts are written.
        """

        resolved_data = data_root or os.environ.get("BUSI_ROOT") or PROJECT_ROOT / "data" / "BUSI"
        resolved_output = (
            output_root
            or os.environ.get("PROJECT_OUTPUT_ROOT")
            or PROJECT_ROOT / "results" / "runs"
        )
        return cls(
            data_root=Path(resolved_data).expanduser().resolve(),
            output_root=Path(resolved_output).expanduser().resolve(),
            seed=int(seed if seed is not None else os.environ.get("PROJECT_SEED", "42")),
        )
