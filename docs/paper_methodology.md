# Paper methodology and implementation mapping

## Research context

The paper is titled **Breast Cancer Detection Using VGG16 Deep Learning Model** and the project brief identifies Kummari Vishnu, Chegu Bhargav Srikar, Pagalla Ashok, and supervisor Lakhan Dev Sharma of VIT-AP University. The paper PDF remains the authoritative source for exact wording and citations; it should not be edited as part of this code repository.

## Paper-faithful binary baseline

The primary workflow uses the BUSI benign and malignant classes only. The normal class is excluded. For a supplied BUSI root, the loader locates `benign`/`benign_tumor` and `malignant`/`malignant_tumor`, ignores mask files, and errors clearly if either class is missing.

The paper describes grayscale images repeated into three channels. The baseline therefore loads a grayscale plane using OpenCV and explicitly repeats it into RGB channel order. This is distinct from the archived notebook's `PIL.Image.convert("RGB")`, which converts color images to RGB rather than first forcing grayscale. Pixel values are scaled to `[0, 1]` by dividing by 255; no ImageNet `preprocess_input` transform is applied.

The image is directly resized to 224 × 224 for the baseline. The aspect-ratio-preserving letterbox transform is a separate experiment and does not replace the baseline.

Labels are integers internally and converted to two-class one-hot vectors in the TensorFlow input pipeline. The requested stratified 80:20 split is created first. Because validation is needed for early stopping and model checkpointing, 10% of the 80% training portion is reserved for validation. With 647 images, the archived split recorded 465 train, 52 validation, and 130 test images (about 72%/8%/20% of the full subset). The code records exact split counts per run.

The baseline architecture is ImageNet-pretrained VGG16 with `include_top=False`, a frozen convolutional base, GlobalAveragePooling2D, Dense(256, ReLU), Dropout(0.5), and Dense(2, softmax). It uses Adam at 1e-4, categorical cross-entropy, batch size 16, up to 20 epochs, EarlyStopping on validation loss with patience 5, and a best-validation-loss checkpoint.

## Methodology differences and uncertainty

- The project brief reports a paper ROC-AUC of 0.67. Notebook-measured values are separate and must not be substituted for the paper's result.
- The project brief notes wording inconsistencies in the paper about freezing versus fine-tuning. The exact passages could not be verified from the PDF in this environment. The baseline follows the explicitly described frozen-base method; fine-tuning is a separate experiment.
- The project brief says the BUSI ZIP used for the experiments does not contain separate mask files, despite masks being mentioned in the paper. Classification in this repository does not consume masks. Confirm the exact BUSI release and archive contents before making a stronger dataset claim.
- The existing notebook split is stratified by image, not by patient or lesion. Related images may cross split boundaries. The implementation preserves that existing protocol for faithful comparison and records `grouped_by_case: false`; this is a limitation, not a recommended final study design.

