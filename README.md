# Fairness-Aware Multimodal Machine Unlearning

This repository contains the code developed for the Master's thesis **"Fairness-Aware Multimodal Machine Unlearning"**. The project investigates machine unlearning methods for multimodal architectures while evaluating both predictive performance and fairness-related metrics.

## Repository Structure

```text
.
├── notebooks/
│   ├── dataset_analysis_preprocessing/
│   ├── evaluate_models/
│   ├── train_models/
│   └── unlearning/
├── json_maps/
├── utils/
├── checkpoints/
├── dataset-orig/
├── dataset_resized_stratified/
└── mal_ben_dataset_resized_stratified/
```

<details>
<summary><b>notebooks/</b></summary>

Contains the Jupyter notebooks used throughout the thesis for data preparation, model training, evaluation, and machine unlearning experiments.

### dataset_analysis_preprocessing

Contains notebooks for:

- Analysing dataset distributions.
- Creating stratified train, validation, and test splits.
- Preparing datasets for both **11-class** and **binary (malignant/benign)** classification tasks.
- Generating cropped, squared, and resized image versions to reduce storage and computational requirements.

### evaluate_models

Contains notebooks used to evaluate:

- Trained uni-modal models.
- Trained multimodal models.
- Models obtained after applying machine unlearning methods.

### train_models

Contains notebooks for training:

- Uni-modal architectures.
- Multimodal architectures.

### unlearning

Contains notebooks implementing the machine unlearning approaches evaluated in this thesis.

</details>

<details>
<summary><b>json_maps/</b></summary>

Stores JSON mapping files used throughout the project, including:

- Category name ↔ class index mappings.
- Lesion IDs belonging to train, validation, and test splits.
- Lesion IDs belonging to forget sets.
- Metadata value ↔ one-hot encoding position mappings.
- Separate mappings for multiclass and binary classification tasks.
- Others.
</details>

<details>
<summary><b>utils/</b></summary>

Contains reusable utility notebooks and helper code, including:

- Custom dataset classes.
- Evaluation utilities.
- Uni-modal training utilities.
- Multimodal training utilities.
- Machine unlearning utilities.
- Shared helper methods used across experiments.
- Others.

</details>

<details>
<summary><b>checkpoints/</b></summary>

Placeholder directory used for storing and loading trained model checkpoints.

</details>

<details>
<summary><b>dataset-orig/</b></summary>

Placeholder directory containing the original dataset.

</details>

<details>
<summary><b>dataset_resized_stratified/</b></summary>

Placeholder directory containing the resized and stratified train, validation, and test splits used for the **11-class classification task**.

</details>

<details>
<summary><b>mal_ben_dataset_resized_stratified/</b></summary>

Placeholder directory containing the resized and stratified train, validation, and test splits used for the **binary malignant vs. benign classification task**.

</details>
