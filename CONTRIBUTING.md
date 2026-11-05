# Contributing

Contributions should keep the research workflow reproducible, the historical results traceable, and all medical data outside the repository.

## Set up a development environment

Use Python 3.10–3.12. Create a virtual environment, then install `requirements.txt` for work that needs TensorFlow or the full application dependencies. Static tests do not require those packages. Install Ruff separately when working on code quality:

```bash
python -m pip install -r requirements-dev.txt
```

## Validate changes

From the repository root, run:

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests scripts
ruff check src tests scripts
ruff format --check src tests scripts
```

The unittest suite parses each public notebook's JSON and Python code cells without executing notebook code or importing TensorFlow. Keep notebook outputs cleared before submitting.

## Code and experiment expectations

- Keep the paper-faithful baseline distinct from extensions; do not silently change preprocessing, split logic, architecture, or hyperparameters.
- Add new experiment names and settings explicitly, and document the exact split, dataset subset, threshold source, and any model-selection use of validation data.
- Preserve historical values in `results/metrics/model_comparison.csv`; add new results as clearly labeled records with the evaluation split and caveats.
- Do not claim an independent test result when a test split was reused or when validation informed model selection.
- Keep imports, path handling, and tests compatible with Windows and Ubuntu where practical. Do not make static tests import TensorFlow-dependent modules.
- Use the Ruff rules and formatting configured in `pyproject.toml`.

## Data privacy and reproducibility

Never commit BUSI or BUS-BRA images, masks, derived scans, patient information, datasets, model checkpoints, or run outputs. Review `git status` and the staged file list before committing. Keep dataset paths and output directories local or outside the repository. Record seeds, split counts, preprocessing, model settings, threshold provenance, and software versions for new runs. Do not run training as part of lightweight CI.

## Pull requests

Open a pull request against `main`, summarize the change and its research implications, list the static checks performed, and identify any runtime checks that were not run. Include result figures only when generated from tracked project metrics and clearly labeled with their evaluation split. Do not include new medical data or personal information in code, notebook outputs, screenshots, or issue attachments.
