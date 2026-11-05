# Experimental findings

## Interpretation rules

The metrics below were recorded in the existing Colab notebook and project audit notes. No result was regenerated during repository implementation. The BUSI test split was evaluated repeatedly across experiments, so all BUSI test figures are exploratory. They are not an untouched final test evaluation and do not establish clinical performance. The paper-reported ROC-AUC of 0.67 is a separate reported value, not a notebook-measured result.

The new augmentation extension uses horizontal flips, `RandomRotation(0.05)` (approximately ±18° because Keras expresses this factor as a fraction of a full turn), and `RandomZoom(0.10)`. These are implementation settings for the new extension, not paper-specified methods or historical results.

The 93.98% value is training accuracy. The fine-tuned experiment's held-out BUSI test accuracy was 86.92%. The letterbox experiment has validation metrics only. BUS-BRA is reported separately as an external dataset evaluation; its lower performance indicates a generalization challenge.

## Recorded results

The full machine-readable comparison is in [`../results/metrics/model_comparison.csv`](../results/metrics/model_comparison.csv). Confusion matrices use class order benign, malignant.

| Experiment / split | Accuracy | ROC-AUC | Malignant precision | Malignant recall | Malignant F1 | Additional details |
|---|---:|---:|---:|---:|---:|---|
| Original VGG16, BUSI exploratory test, default argmax | 76.15% | 0.8003 | 78.95% | 35.71% | 49.18% | Confusion matrix `[[84, 4], [27, 15]]`; 130 images |
| Original VGG16, threshold 0.3268, BUSI exploratory test | 73.85% | 0.8003 | 57.41% | 73.81% | 64.58% | Threshold selected on BUSI validation |
| Class-weighted VGG16, threshold 0.4872, BUSI exploratory test | 75.38% | 0.8036 | 59.62% | 73.81% | 65.96% | Confusion matrix `[[67, 21], [11, 31]]` |
| Class-weighted VGG16 with block-5 fine-tuning, threshold 0.6782, BUSI exploratory test | 86.92% | 0.9351 | 83.78% | 73.81% | 78.48% | Train accuracy 93.98%; final-epoch validation accuracy 84.62%; best validation loss at epoch 9; confusion matrix `[[82, 6], [11, 31]]` |
| Fine-tuned BUSI model, BUS-BRA image level | 64.00% | 0.6232 | 43.00% | 34.43% | 38.24% | 1,875 images |
| Fine-tuned BUSI model, BUS-BRA case level | 68.80% | 0.6475 | 52.75% | 28.07% | 36.64% | 1,064 cases; average image probabilities per case |
| Letterbox preprocessing with two-stage training, BUSI validation | 82.69% | 0.8689 | Not recorded | Not recorded | Not recorded | Validation-only; 647 images; class weighting and block-5 fine-tuning also used |

## Comparability and caveats

- Test accuracy depends on the decision threshold. The original default-argmax result (76.15%) and its validation-selected-threshold result (73.85%) use different operating points; ROC-AUC remains 0.8003.
- The original notebook's saved `model_comparison.csv` compared only the original thresholded model and class-weighted model. This documentation table includes the later fine-tuned and letterbox results from notebook outputs and the project audit notes.
- Letterbox is confounded with class weighting and block-5 fine-tuning; it does not isolate the effect of padding.
- Repeated test-set evaluation informed experiment comparison. A future confirmatory result needs a fresh, untouched test set or a grouped/nested evaluation design.
- BUS-BRA thresholding in the recorded experiment was locked using BUSI validation data. Do not tune a threshold on BUS-BRA and then call the same evaluation independent.
- The external result is image- and case-level research evaluation, not clinical validation.

