# Glaucoma Detection from Retinal Images: Deep Learning vs Classical Machine Learning

**Author:** Mayishat Altab (GitHub: [@mayishataltab](https://github.com/mayishataltab)

## Overview

This project classifies retinal fundus images as **glaucoma** or **normal**. It compares two transfer-learning models (EfficientNetB3 and DenseNet121) with two classical machine-learning models (SVM and XGBoost) trained on hand-crafted image features. It also uses Grad-CAM to check whether the deep models look at the retina or at the black image background.



> This is a research and learning project. It is not a clinical tool and must not be used for medical diagnosis.

## Dataset

- 6,000 retinal images: 3,000 glaucoma and 3,000 normal (balanced).
- Stratified split with seed 42: **4,800 train / 600 validation / 600 test**, with 300 glaucoma and 300 normal images in the validation and test sets.


## Methods

**Deep learning (EfficientNetB3, DenseNet121)**
- ImageNet-pretrained backbone, global average pooling, dropout 0.3, 2-class softmax output.
- Input 224 x 224, batch size 32, mixed precision.
- Augmentation (training only): horizontal flip, small rotation, zoom, contrast.
- Stage 1: train the new head with the backbone frozen (up to 10 epochs, learning rate 1e-3).
- Stage 2: unfreeze and fine-tune (up to 30 epochs, learning rate 1e-4).
- Early stopping and checkpointing on validation AUC, with learning-rate reduction on plateau. The best epoch was 22 for EfficientNetB3 and 7 for DenseNet121.

**Classical machine learning (SVM, XGBoost)**
- Images resized to 128 x 128. Features (322 per image): HOG on grayscale, uniform LBP histogram, and per-channel 32-bin color histograms.
- SVM (RBF kernel): randomized search, 5-fold cross-validation, best C = 10, gamma = scale. Final model retrained on train + validation.
- XGBoost: randomized search, 3-fold cross-validation, best max depth 6, learning rate 0.1, subsample 0.8, column sample 0.8.

**Grad-CAM audit**
- Grad-CAM heatmaps for all 600 test images per deep model. An image is flagged when the strongest activation falls mostly on the dark background or border instead of the retina, which can indicate shortcut learning.

## Results (held-out test set, 600 images)

| Model | Accuracy | AUC | Glaucoma precision | Glaucoma recall | Inference time per image |
|---|---|---|---|---|---|
| DenseNet121 | 0.9000 | 0.9650 | 0.8871 | 0.9167 | 0.0435 s |
| EfficientNetB3 | 0.8500 | 0.9334 | 0.8750 | 0.8167 | 0.1025 s |
| XGBoost | 0.7450 | 0.8223 | n/a | n/a | 0.000015 s |
| SVM (RBF) | 0.7350 | 0.8287 | n/a | n/a | 0.0011 s |

- DenseNet121 gave the best results, and both deep models clearly outperformed the classical models.
- Inference times for SVM and XGBoost exclude feature extraction, so they are not directly comparable with the deep models, which include image loading and preprocessing.
- The precision, recall, and F1 in the one-line test summaries printed in the notebook are for label 1, which is the **normal** class. The per-class glaucoma values above come from the notebook's classification reports.

### Grad-CAM audit

| Model | Images flagged for background attention | Accuracy when flagged | Accuracy when not flagged |
|---|---|---|---|
| DenseNet121 | 1.5% | 77.8% | 90.2% |
| EfficientNetB3 | 9.3% | 57.1% | 87.9% |

EfficientNetB3 relies on the image background more often, and its accuracy is much lower on those images. DenseNet121 focuses on the retina more consistently.


## Tools

Python 3.12, TensorFlow/Keras, scikit-learn, XGBoost, scikit-image, pandas, NumPy, Matplotlib, seaborn, joblib.
