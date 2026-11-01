# Glaucoma Detection from Retinal Fundus Images: Deep Learning vs Classical Machine Learning

**Author:** Mayishat Altab Mridu ([@mayishataltab](https://github.com/mayishataltab))
Department of Computer Science and Engineering, Daffodil International University, Dhaka, Bangladesh

## Overview

This project classifies retinal fundus images as **glaucoma** or **normal**. It compares two transfer-learning models (EfficientNetB3 and DenseNet121) with two classical machine-learning models (SVM and XGBoost) trained on hand-crafted image features. A Grad-CAM audit checks whether the deep models base their decisions on the retina or on the black image background. The best model (DenseNet121) is deployed in **GlaucomaDetect**, a Flask web app for image upload, prediction and PDF reports.

> This is a research project. It is not a clinical tool and must not be used for medical diagnosis.

## Dataset

- [EyePACS-AIROGS-light](https://doi.org/10.34740/KAGGLE/DS/3222646) (Kaggle), derived from the AIROGS challenge dataset.
- 6,000 retinal images: 3,000 glaucoma and 3,000 normal (balanced). Exact-duplicate check (MD5): no duplicates found.
- Stratified split with seed 42: **4,800 train / 600 validation / 600 test**, balanced in every subset.
- The same split is used for all four models.

## Methods

**Deep learning (EfficientNetB3, DenseNet121)**
- ImageNet-pretrained backbone, global average pooling, dropout 0.3, single sigmoid output (binary cross-entropy).
- Input 224 × 224 with raw 0–255 pixels; the ImageNet normalization is built into each model, so training and deployment use identical preprocessing.
- Augmentation (training set only): horizontal flip, rotation, zoom, contrast.
- Stage 1 (warm-up): backbone frozen, only the head is trained (10 epochs, Adam, learning rate 1e-3).
- Stage 2 (fine-tuning): whole network trained (up to 30 epochs, learning rate 1e-4).
- Early stopping (patience 6) and checkpointing on validation AUROC, learning-rate halving on plateau, mixed precision, batch size 32.
- Best fine-tuning epoch: 8 for EfficientNetB3, 12 for DenseNet121.

**Classical machine learning (SVM, XGBoost)**
- Images resized to 128 × 128. Features (322 per image): HOG on grayscale (216), uniform LBP histogram (10), and 32-bin color histograms per RGB channel (96).
- SVM (RBF kernel): grid search with 5-fold cross-validation on the training set; best C = 10, gamma = 0.001; standardized features.
- XGBoost: randomized search (20 settings) with 5-fold cross-validation on the training set; best 800 trees, max depth 4, learning rate 0.1, subsample 0.8, column sample 0.7, min child weight 3.
- Both models are trained on the same 4,800 training images as the deep models.

**Evaluation**
- Accuracy, precision, recall, specificity, F1-score and AUROC on the held-out test set, with 95% bootstrap confidence intervals (1,000 resamples).
- McNemar's exact test for pairwise model comparisons.
- Glaucoma is the positive class (label 1).

**Grad-CAM audit**
- Grad-CAM heatmaps (computed in float32 from the last convolutional layer) for all 600 test images per deep model.
- A field-of-view mask separates the retina from the black border. An image is flagged when more than 15% of its strongest activation (top 10% of pixels) falls outside the retina.

## Results (held-out test set, 600 images)

| Model | Accuracy (95% CI) | Precision | Recall | Specificity | F1 | AUROC (95% CI) |
|---|---|---|---|---|---|---|
| **DenseNet121** | **0.9083** (0.883–0.932) | 0.8990 | **0.9200** | 0.8967 | **0.9094** | **0.9674** (0.954–0.980) |
| EfficientNetB3 | 0.8583 (0.828–0.887) | 0.8881 | 0.8200 | 0.8967 | 0.8527 | 0.9418 (0.922–0.960) |
| XGBoost | 0.7400 (0.707–0.773) | 0.7368 | 0.7467 | 0.7333 | 0.7417 | 0.8230 (0.791–0.853) |
| SVM (RBF) | 0.7183 (0.683–0.753) | 0.7191 | 0.7167 | 0.7200 | 0.7179 | 0.8082 (0.773–0.842) |

- DenseNet121 gave the best results. It significantly outperformed EfficientNetB3 (McNemar p < 0.001), and both deep models significantly outperformed both classical models (p < 10⁻⁸). XGBoost vs SVM was not significant (p = 0.198).
- DenseNet121 confusion matrix: 276 true positives, 269 true negatives, 24 false negatives, 31 false positives.

### Inference time (per image, end-to-end from image file to prediction, batch size 1)

| Model | Hardware | Time per image | Classifier only |
|---|---|---|---|
| DenseNet121 | GPU (Tesla T4) | 18.9 ms | – |
| EfficientNetB3 | GPU (Tesla T4) | 17.0 ms | – |
| XGBoost | CPU | 13.4 ms | 0.5 ms |
| SVM | CPU | 14.9 ms | 1.5 ms |

For SVM and XGBoost, most of the time is spent on hand-crafted feature extraction, which is included so that all four models are timed end-to-end.

### Grad-CAM audit

| Model | Mean background attention | Images flagged | Accuracy when flagged | Accuracy when not flagged |
|---|---|---|---|---|
| DenseNet121 | 0.020 | 3.0% (18 of 600) | 72.2% | 91.4% |
| EfficientNetB3 | 0.021 | 3.5% (21 of 600) | 57.1% | 86.9% |

Both models focus mainly on the optic disc region. Images flagged for background attention are classified much less accurately, so the flag is a useful warning sign for unreliable predictions. Many of the highest-scoring cases have the optic disc near the image edge, where the coarse 7 × 7 Grad-CAM map spills over the retina border.

## Web app (GlaucomaDetect)

- Flask app that loads the trained DenseNet121 model (`.keras`, Keras 3), validates the upload, preprocesses it exactly as in training (PIL, RGB, 224 × 224, raw 0–255 pixels) and returns the label, a confidence score (capped at 99.9%) and a downloadable PDF report.
- Uploaded images are deleted from the server after the prediction and report are generated.
- Model inference takes about 0.4 s per image on a standard CPU.
- The trained model file is not stored in this repository because of GitHub's file-size limit; place it in `static/models/` before running the app locally.

## Reproducing the experiments

1. Open the notebook on Kaggle, add the EyePACS-AIROGS-light dataset as input, and enable a GPU accelerator and internet access.
2. Run all cells (about 1–1.5 hours on a Tesla T4). The final cell prints a summary of all results and saves the figures, tables and trained models.

GPU training is not bit-exact reproducible, so small run-to-run differences in the deep-learning results are expected.

## Tools

Python, TensorFlow 2.18 / Keras 3.8, scikit-learn, XGBoost 2.0, scikit-image, SciPy, pandas, NumPy, Matplotlib, seaborn, joblib, Flask, Pillow, ReportLab.

## Citation

If you use this work, please cite the accompanying paper (under review at *SN Computer Science*):

> Mridu MA, et al. Comparative evaluation of machine learning and transfer-learning-based deep learning models for explainable glaucoma detection from retinal fundus images. Manuscript under review, 2026.
