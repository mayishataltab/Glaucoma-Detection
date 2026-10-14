# Automated Glaucoma Detection from Retinal Fundus Images — ML vs. DL

**Author:** Mayishat Altab

A comparative machine learning and deep learning project for binary classification of retinal fundus images into glaucoma and normal categories. This project evaluates traditional machine learning classifiers and transfer-learning architectures, with additional Grad-CAM visualization and background-attention analysis.

## Overview

Glaucoma is an eye disease that can damage the optic nerve and cause vision loss. This project explores automated classification of retinal fundus images as a research application of machine learning and medical image analysis.

The notebook compares four approaches:

- **EfficientNetB3** — transfer-learning-based convolutional neural network.
- **DenseNet121** — transfer-learning-based convolutional neural network.
- **Support Vector Machine (SVM)** — traditional classifier using handcrafted image features.
- **XGBoost** — gradient-boosted decision tree classifier using handcrafted image features.

In addition to classification, the notebook investigates model behavior using Grad-CAM heatmaps and a heuristic audit of attention falling on dark image backgrounds or borders.

> **Research disclaimer:** This project is for educational and research purposes. It is not a clinically validated diagnostic system and must not be used as a substitute for professional medical assessment.

## Key Features

- Binary glaucoma-versus-normal image classification.
- Stratified train, validation, and test splitting.
- Image resizing and preprocessing.
- Data augmentation for deep learning.
- Transfer learning and fine-tuning with EfficientNetB3 and DenseNet121.
- Handcrafted feature extraction using Histogram of Oriented Gradients (HOG) and Local Binary Patterns (LBP).
- SVM and XGBoost model training and tuning.
- Classification metrics and confusion-matrix visualization.
- Grad-CAM heatmap generation for deep learning models.
- Background-attention auditing to investigate potential shortcut learning.
- Export of model checkpoints, training histories, evaluation reports, and visualizations.

## Dataset

The notebook is configured for a dataset of approximately 6,000 retinal fundus images, with approximately 3,000 images in each class, according to the notebook's description.

Expected directory structure:

```text
Glaucoma_Dataset2/
├── glaucoma/
│   ├── image_001.jpg
│   └── ...
└── normal/
    ├── image_001.jpg
    └── ...
```

The dataset is not included in this repository. Obtain it from an authorized source and follow the relevant access, licensing, and redistribution conditions.

**Important:** Verify the actual image counts and class distribution when running the notebook; the stated counts should not be treated as independently verified dataset statistics.

## Experimental Setup

The notebook uses a stratified split:

| Partition | Proportion |
|---|---:|
| Training | 80% |
| Validation | 10% |
| Testing | 10% |

A fixed random seed is used to support reproducibility.

The deep learning models use resized 224 × 224 RGB images. Training augmentation includes horizontal flipping, rotation, zoom, and contrast adjustment.

## Models and Methods

### 1. EfficientNetB3

An ImageNet-pretrained EfficientNetB3 backbone is used with a global average pooling layer, dropout, and a two-class softmax output layer.

The training pipeline includes a warm-up stage followed by fine-tuning. Model checkpoints and early-stopping callbacks monitor validation AUC.

### 2. DenseNet121

DenseNet121 uses a similar transfer-learning approach, with a global average pooling layer, dropout, and a two-class output layer. Its performance is evaluated on the held-out test partition.

### 3. Support Vector Machine (SVM)

The SVM uses an RBF kernel and handcrafted image features, including HOG and LBP descriptors. Hyperparameter search is used to identify a suitable configuration.

### 4. XGBoost

XGBoost is trained on the handcrafted image-feature representation. The notebook includes randomized hyperparameter search and classification evaluation.

## Evaluation

The notebook includes code for comparing the models with:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC where applicable
- Classification reports
- Confusion matrices
- Training and inference timing where recorded

Final performance claims should be based on actual predictions on the held-out test set. This README does not assert a particular best-performing model or numeric test score without a verified result from the notebook's execution.

## Explainability: Grad-CAM

Gradient-weighted Class Activation Mapping (Grad-CAM) generates heatmaps that highlight image regions associated with the predictions of EfficientNetB3 and DenseNet121.

The heatmaps can help researchers inspect model behavior, but they do not prove that the model uses medically valid features or that its predictions are clinically correct.

## Background-Attention Audit

The notebook includes an additional audit that estimates how much of the high-activation Grad-CAM region falls outside the foreground retinal field of view.

The analysis:

1. Generates Grad-CAM heatmaps for test images.
2. Estimates a foreground mask using image brightness.
3. Calculates a background-attention fraction.
4. Flags cases that exceed a selected threshold.
5. Summarizes results by model and true class.
6. Visualizes examples with high background attention.

This is a heuristic diagnostic for possible shortcut learning. A flagged case does not necessarily prove that a model relied on background artifacts, and an unflagged case does not guarantee robust behavior.

## Technologies Used

- Python
- TensorFlow and Keras
- Scikit-learn
- XGBoost
- NumPy and Pandas
- scikit-image
- Matplotlib and Seaborn
- Joblib

## Getting Started

### Requirements

Use Python 3.10 or another version compatible with your chosen TensorFlow release. A GPU is recommended for training the deep learning models, but a compatible CPU environment may also be used.

### Install dependencies

In Google Colab or a compatible notebook environment:

```python
%pip install tensorflow scikit-image xgboost joblib
```

The notebook also imports NumPy, Pandas, scikit-learn, Matplotlib, and Seaborn. Install these packages if they are not already available.

For reproducibility, document and pin the dependency versions used in your final experiments.

### Configure the dataset path

Update the `DATASET_DIR` variable in the dataset-configuration cell:

```python
DATASET_DIR = "/path/to/Glaucoma_Dataset2"
```

Replace the placeholder with the actual dataset location. Ensure that the directory contains the expected `glaucoma` and `normal` folders.

The original notebook uses a Kaggle-specific path, so the path must be changed when running the notebook in Google Colab or a local environment.

### Run the notebook

1. Open the notebook in Google Colab or Jupyter Notebook.
2. Install the dependencies.
3. Configure the dataset path.
4. Run the cells in order.
5. Inspect the training histories, test-set metrics, and confusion matrices.
6. Review the Grad-CAM heatmaps and background-attention audit.
7. Examine the generated reports and visualizations.

The complete workflow may require substantial processing time, especially when training both deep learning models and running the attention audit over the full test set.

## Generated Outputs

The notebook writes artifacts to `glaucoma_outputs_final/`. Depending on which cells complete successfully, outputs may include:

- Trained Keras models and checkpoints.
- Training-history CSV files.
- Training, validation, and test path manifests.
- Model evaluation reports.
- Confusion-matrix plots.
- Grad-CAM visualizations.
- Background-attention audit summaries.
- CSV reports and other diagnostic figures.

Generated artifacts are not necessarily included in this repository.

## Limitations

- Results depend on the dataset, preprocessing, split strategy, and software environment.
- A random image-level split does not establish generalization to independent patients or external clinical populations.
- Grad-CAM is a visualization technique, not proof of causal reasoning.
- Background-attention estimates depend on the foreground-mask heuristic and chosen threshold.
- External validation and clinical assessment are required before considering real-world medical use.
- Reproducibility requires access to the dataset and compatible dependencies.

## Future Work

- Evaluate on an independent external dataset.
- Use patient-level splitting when patient identifiers are available.
- Report confidence intervals and statistical comparisons between models.
- Investigate model robustness to image artifacts and acquisition differences.
- Evaluate explanations with domain experts.
- Develop a research demonstration interface for image predictions.

## Author

**Mayishat Altab**

This repository documents my work on machine learning and deep learning for retinal fundus image classification, including comparative model evaluation and exploratory explainability analysis.

**Research interests:** Machine Learning, Deep Learning, Medical Image Analysis, and Explainable AI.

**GitHub:** [Your GitHub Profile](https://github.com/mayishataltab)

Replace the placeholder above with your GitHub username.

## License

Choose a license for the original code before distributing it. Dataset usage and redistribution rights are governed separately by the dataset provider's terms.

## Responsible Use

This repository is intended for research and educational use. Predictions from these models should not be used to diagnose, exclude, or treat glaucoma in individual patients.
