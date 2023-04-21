# DeepKinZero

This repository contains the Pytorch implementation for the paper: Iman Deznabi, Busra Arabaci, Mehmet Koyutürk, Oznur Tastan, DeepKinZero: Zero-Shot Learning for Predicting Kinase-Phosphosite Associations Involving Understudied Kinases, Bioinformatics, , btaa013, https://doi.org/10.1093/bioinformatics/btaa013

create conda environment:
conda env create --name NAME --file environment.yml

## Train DeepKinZero
```
python main.py --MODE train --DEVICE cuda --TRAIN_DATA train_data_path --VAL_DATA val_data_path --VAL_KINASE_CANDIDATES val_kinase_path
```

## Test DeepKinZero
```
python main.py --MODE test --DEVICE cuda --TEST_DATA test_data_path --TEST_KINASE_CANDIDATES test_kinase_path
```