# DeepKinZero

This repository contains the Pytorch implementation for the paper: Iman Deznabi, Busra Arabaci, Mehmet Koyutürk, Oznur Tastan, DeepKinZero: Zero-Shot Learning for Predicting Kinase-Phosphosite Associations Involving Understudied Kinases, Bioinformatics, , btaa013, https://doi.org/10.1093/bioinformatics/btaa013

## File Structure

**main.py:** Main file to run the project. Creates datasets, models, optimizers, trainers. Then, train models and print train, validation and test results.

**model.py:** Bidirectional LSTM model. This is for phosphosite embeddings.

**trainer.py:** Trainer class that train the models and make predictions on given validation or test data.

**dataset.py:** Dataset class contains the data that model get as input.

**create_dataset.py:** Creates datasets for train, validation and test. Called in **main.py**.

**train_utils.py:** Some util functions to be used on training or evaluation.

**utils.py:** Other util functions.

**config.py:** All configuration parameters e.g. hyperparameters, data paths

**data/amino_acids.py:** Amino Acid Class

**data/kinase_embeddings.py:** Kinase Embedding class contains all of the kinase informations

**data/sequence_data.py:** This class contains all phosphosite and kinase related informations.
