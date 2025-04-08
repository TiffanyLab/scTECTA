# scTECTA: Asymmetric Deep Transfer Learning for Cross-Patient Tumor Microenvironment Single-Cell Annotation

<img src="https://github.com/TiffanyLab/scTECTA/blob/main/Figure_scTECTA.png" width="900">

scTECTA is a deep transfer learning model designed to transfer single-cell annotation information from an annotated patient dataset to an unannotated patient dataset, adapted for the tumor microenvironment. This project is a sample implementation based on `torch`.

## Requirements

+ python >= 3.8
+ torch >= 2.4.0
+ numpy >= 1.24.3
+ pandas >= 2.0.3
+ scanpy >= 1.9.8
+ scikit-learn >= 0.24.2

## Usage
### Step 1: Prepare data
You need to place the sample file in the `./Datasets` folder of the project root directory as per the instructions below, or organize your own data according to the format of the sample file.

### Step 2: Training the model
```
python train.py --use_pca False --n_epochs 2000 --dense_dim 50 --hidden_dim 256 --weight 0.1 --target_pnum 2 --source_disease train --target_disease test --dropout_ratio 0.1 --k 0.01
```

## Datasets
You can download the sample dataset from Google Drive at the following link: https://drive.google.com/drive/folders/1Llmpfm3_ndEOiSO5YtB2uUuxv5bX06Q7?usp=drive_link

The `train_matrix.csv` file is the single-cell expression matrix used for training, `train_label.csv` contains the cell type labels for the training data, `test_matrix.csv` is the single-cell expression matrix used for testing, and `test_label.csv` contains the cell type labels for the test data. After downloading, place `train_matrix.csv` and `train_label.csv` in the `./Datasets/train` folder, and place `test_matrix.csv` and `test_label.csv` in the `./Datasets/test` folder.
