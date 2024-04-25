![arch](https://github.com/idrugLab/DualPG-DTA/blob/main/arch.png)
<p align="center">Architecture diagram of DualPG-DTA</p>
<p align="center">(a) drug feature extraction module; (b) protein feature extraction module; (c) fusion prediction module.</p>

### Requirements
This project is developed using python 3.8.19, and mainly requires the following libraries.
```
biopython==1.83
fair-esm==2.0.0
networkx==3.0
numpy==1.24.1
optuna==3.6.1
pandas==2.0.3
pytorch-lightning==2.2.3
rdkit==2023.9.5
scikit-learn==1.3.2
scipy==1.10.1
torch==2.2.2+cu118
torch_geometric==2.5.2
torchmetrics==1.3.2
```
To install [requirements](https://github.com/idrugLab/DualPG/blob/main/requirements.txt):
```
pip install -r requirements.txt
```

### Files & Folders Description
- **run.py**: Source code for using the model. You will only use this file if you do not need to retrain the model.
- **main.py**: Source code for the model training process, including model training, persistence of weights, and output of evaluation metrics.
- **metrics.py**: Source code for evaluation metrics implementation.
- **models.py**: Source code for model architecture implementation.
- **preprocess.py**: Source code for data preprocessing, including building molecular graphs and protein graphs.
- **dataset**: Datasets (Davis and KIBA) for training, validation and testing.
- **pts**: Where the model's weights are saved.

### Usage
If binding affinity prediction is performed for one drug and one target, use the following code:
```
python run.py --drug <drug_smiles> --target <target_sequence>
```
Example:
```
python run.py --drug "CC(C)c1nc(CN...C)cs1" --target "MAPLYL...VDGTVSGA"
```
Also, you can make predictions for a large set of drugs and targets at the same time, using the following code:
```
python run.py --input <input_csv_filename>
```
Example:
```
python run.py --input "2024.csv"
```
And the output filename will be named `out_2024.csv`.
