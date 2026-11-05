# Single-image inference

`src.predict` runs inference with a **locally saved** Keras checkpoint. It does not train or fine-tune a model and does not download a checkpoint or ImageNet weights. The prediction is a research output, not a medical diagnosis.

## Prerequisites

Use Python 3.10–3.12 in an environment with the project dependencies installed:

```bash
python -m pip install -r requirements.txt
```

OpenCV and TensorFlow are required at runtime. Install a TensorFlow build compatible with your operating system and hardware if the pinned project requirements are not compatible with your environment.

## Obtain a checkpoint

Use `best_model.keras` from an existing completed training run, usually under `results/runs/<run-name>/best_model.keras`. Keep the run's `metrics.json` beside it if you want to use that checkpoint's BUSI validation-selected threshold. Checkpoints and ultrasound images are local artifacts and should not be added to GitHub.

The inference preprocessing reuses `src.preprocessing.preprocess_grayscale_image` with its research default: grayscale read, resize to 224 × 224, three identical channels, and pixel scaling to `[0,1]`.

## Run locally

From the repository root, use argmax classification by default:

```bash
python -m src.predict --model results/runs/<run-name>/best_model.keras --image /path/to/ultrasound.png
```

Inference uses CPU by default. To choose a malignant probability threshold explicitly:

```bash
python -m src.predict --model results/runs/<run-name>/best_model.keras --image /path/to/ultrasound.png --threshold 0.42
```

To use the threshold saved for that run, pass the matching `metrics.json` from the same run directory:

```bash
python -m src.predict --model results/runs/<run-name>/best_model.keras --image /path/to/ultrasound.png --metrics-json results/runs/<run-name>/metrics.json
```

Save the prediction and its metadata as JSON with `--output-json`:

```bash
python -m src.predict --model results/runs/<run-name>/best_model.keras --image /path/to/ultrasound.png --output-json results/predictions/prediction.json
```

The output reports benign and malignant probabilities, predicted class, confidence, decision method, and threshold provenance. It does not include a diagnosis.

## Run in Google Colab

1. Open a Colab notebook and mount Google Drive if the checkpoint and image are stored there.
2. Clone or upload this repository into the runtime, then change to its project root.
3. Install dependencies in the Colab runtime with `!pip install -r requirements.txt` (choose a TensorFlow/OpenCV combination compatible with the active Colab runtime if necessary).
4. Run the command below, replacing the paths with your Drive or runtime paths:

```bash
!python -m src.predict --model "/content/drive/MyDrive/path/to/best_model.keras" --image "/content/drive/MyDrive/path/to/ultrasound.png" --device cpu
```

Add `--metrics-json` only when using the `metrics.json` from that checkpoint's run folder. Colab inference also uses CPU; the code does not start training or fine-tuning.

## Threshold provenance

- With neither `--threshold` nor `--metrics-json`, the class is selected by argmax over the two class probabilities.
- `--threshold` is an explicit malignant-probability cutoff and must be between 0 and 1 inclusive. Its validation provenance is the caller's responsibility.
- `--metrics-json` must point to a metrics file in the checkpoint's run directory. The file must identify BUSI validation as its evaluation split and validation data as the threshold-selection source. The threshold is never taken from historical project metrics or an unrelated directory.
- Supply either `--threshold` or `--metrics-json`, not both.

Example JSON structure (illustrative values only):

```json
{
  "predicted_class": "benign",
  "confidence": 0.82,
  "probabilities": {
    "benign": 0.82,
    "malignant": 0.18
  },
  "decision_method": "Argmax",
  "threshold": null,
  "threshold_source": "Argmax (no threshold supplied)",
  "model_path": "/local/path/best_model.keras",
  "image_path": "/local/path/ultrasound.png"
}
```

## Limitations and responsible use

This interface performs inference only and assumes the checkpoint has the expected two-class output order: benign, then malignant. Preprocessing follows the project's current resize pipeline; checkpoints trained with a different preprocessing method may not be compatible. Predictions can be wrong and are not validated for clinical use. Do not use them to diagnose, screen, or make treatment decisions. Protect ultrasound images and model artifacts as sensitive data, and consult qualified clinicians for medical interpretation.
