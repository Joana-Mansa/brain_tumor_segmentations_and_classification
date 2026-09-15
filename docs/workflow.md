# VGG16 experiment guide

## Provenance

The notebook comes from this repository’s `deep_learning_` branch. It covers VGG16 feature extraction and fine-tuning and is related to the larger [brain-MRI capstone](https://github.com/Joana-Mansa/machine_deep_learning_project). Saved outputs are historical evidence, not a fresh reproduction.

## Inputs and execution

The original code mounts Google Drive and expects `train_test_dataset` with class-organized training/test image directories. Inspect the data-preparation cells and change `data_dir` to the local directory when not using Colab. Inputs, cohort metadata and trained checkpoints are external.

Read each section in order: data preparation, VGG16 feature extractor, training, evaluation, then fine-tuning. Checkpoint-loading cells need an existing file or a completed training step; do not expect `/content/vgg16_model.pth` to exist on a new machine.

## Validation questions

Confirm patient-level separation and class mappings before rerunning; a random image-level validation split may not preserve patient independence. Check whether modifying validation transforms changes a shared training dataset object. The notebook retains exploratory choices for review rather than claiming a new validated pipeline.

Full training and evaluation were not rerun because the original inputs were unavailable. The recovered notebook was validated as Jupyter format and its cells were inspected; this is not a claim of numerical reproduction.
