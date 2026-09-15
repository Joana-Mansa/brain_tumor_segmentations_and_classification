# 🧠 Brain Tumour Classification: VGG16

An academic MRI classification experiment by Joana Owusu-Appiah, comparing VGG16 as a feature extractor and as a fine-tuned network.

## Open the work

- 📓 [VGG16 notebook](brain_tumor_transfer_learning_vgg16.ipynb): recovered from this repository’s existing `deep_learning_` branch.
- 📚 [Broader brain-MRI capstone](https://github.com/Joana-Mansa/machine_deep_learning_project): preprocessing, segmentation, classical ML and deep learning.
- 📖 [Data and execution guide](docs/workflow.md).

These repositories are related parts of the same academic work. This repository holds the VGG16 experiment; the broader capstone contains the larger workflow.

## Run

```bash
python -m pip install -r requirements.txt
jupyter lab brain_tumor_transfer_learning_vgg16.ipynb
```

The original notebook targets Colab and reads a Google Drive `train_test_dataset` directory. Set the dataset path and skip the Drive mount when running locally. Dataset and checkpoint files are not included. Read the guide before executing the training cells.

## Status

The notebook and historical outputs are now visible on the default branch. The experiments have not been retrained in this maintenance pass. Source changes that affect splits, preprocessing or models require new evaluation before reporting updated scores.
