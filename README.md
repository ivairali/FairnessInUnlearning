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

Placeholder directory containing the original dataset. The Milk10 Benchmark dataset description and download location are available here: https://challenge.isic-archive.com/landing/milk10k/

</details>

<details>
<summary><b>dataset_resized_stratified/</b></summary>

Placeholder directory containing the resized and stratified train, validation, and test splits used for the **11-class classification task**. One of the lession ids, corresponding to a dermoscopic and a normal image is put as an example in each folder. 

</details>

<details>
<summary><b>mal_ben_dataset_resized_stratified/</b></summary>

Placeholder directory containing the resized and stratified train, validation, and test splits used for the **binary malignant vs. benign classification task**. One of the lession ids, corresponding to a dermoscopic and a normal image is put as an example in each folder. 


</details>

## Project overview and results:
### Project Overview

This project investigates the relationship between **machine unlearning** and **fairness** in multimodal healthcare classification models. The experiments evaluate both **sample-level unlearning** (removing training samples) and **feature-level unlearning** (revoking metadata attributes) while analysing their impact on protected demographic subgroups.

The work is motivated by realistic data removal scenarios, such as user deletion requests under GDPR or the revocation of specific data attributes. Using a multimodal skin lesion classification setting, the project examines whether unlearning methods preserve not only predictive performance but also fairness across demographic groups.

### Main Results

- Utility metrics alone do not fully characterise unlearning behaviour.
- Fairness effects can differ substantially from overall performance effects.
- Targeted forgetting of minority subgroups can reveal fairness degradation that is not visible when using random forget sets.
- The effects of feature revocation depend strongly on feature importance and dataset characteristics.
- Fairness should be considered a first-class evaluation criterion for machine unlearning methods.

