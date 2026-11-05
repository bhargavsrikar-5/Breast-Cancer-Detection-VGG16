# Dataset documentation

## BUSI

The Breast Ultrasound Images dataset contains 780 images in the original three-category release: 437 benign, 210 malignant, and 133 normal, according to the project audit notes. The binary classification workflow uses benign and malignant only (647 images). The dataset paper is Al-Dhabyani et al., *Dataset of breast ultrasound images*, Data in Brief 28 (2020), 104863, DOI [10.1016/j.dib.2019.104863](https://doi.org/10.1016/j.dib.2019.104863).

The project audit notes state that the BUSI ZIP used in the historical workflow does not contain separate mask files. The classification pipeline does not require masks. The official dataset publication and the exact downloaded archive should be checked before claiming masks are absent from every BUSI release.

Keep images outside Git. The expected local layout is `data/BUSI/benign_tumor/`, `data/BUSI/malignant_tumor/`, and `data/BUSI/normal_scan/`; `/data/` is ignored. The binary loader reads only the benign and malignant folders. Alternatively set `BUSI_ROOT` or pass `--data-root` to a directory containing the class folders or their parent.

## BUS-BRA

BUS-BRA is used only for external evaluation of a model trained on BUSI. The audit notes record 1,875 images, 1,875 masks, 1,064 cases, 1,268 benign images, 607 malignant images, and BI-RADS categories 2–5. The published dataset article describes 1,875 images from 1,064 patients and standardized 5-fold and 10-fold partitions. See Gómez-Flores, Gregorio-Calas, and Coelho de Albuquerque Pereira, *BUS-BRA: A breast ultrasound dataset for assessing computer-aided diagnosis systems*, Medical Physics 51 (2024), 3110–3123, DOI [10.1002/mp.16812](https://doi.org/10.1002/mp.16812). The dataset is distributed on [Zenodo, DOI 10.5281/zenodo.8231412](https://doi.org/10.5281/zenodo.8231412).

The archived notebook reports that all image IDs matched metadata, all image-mask pairs matched, no dimensions mismatched, labels were consistent within cases, and official folds preserved case grouping. The new external evaluator reads the archive for evaluation and averages image probabilities within case. It does not train on BUS-BRA or select a threshold there. Supply a threshold determined on BUSI validation data.

Do not add BUSI or BUS-BRA files to this repository. A convenient local BUS-BRA path is `data/BUSBRA.zip`, which is ignored along with ZIP files. Review each dataset's current access and use terms at its authoritative distribution site before use or redistribution.

## Splitting and leakage

The paper-reproduction baseline uses a stratified image-level split to match the historical experiment. It does not enforce lesion- or patient-level grouping because matching group identifiers for BUSI have not been established in the local project. Related BUSI images may therefore cross splits. BUS-BRA official folds are case-grouped, but the recorded external test evaluates all images and aggregates cases; it does not use those folds for training.

