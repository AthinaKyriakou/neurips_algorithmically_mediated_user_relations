# Algorithmically Mediated User Relations: Exploring Data's Relationality in Recommender Systems

Repository for the paper "Algorithmically Mediated User Relations: Exploring Data’s Relationality in Recommender Systems" to be presented in the [Regulatable ML Workshop @ NeurIPS 2023](https://regulatableml.github.io).

## About the Project

You can read the submitted paper [here]().

## Installation
To install the needed dependencies run:
- `conda env create -f requirements.yml`
- `conda activate user_intercon_env`

To use matrix factorization algorithms:
- `conda install -c anaconda cython`
- Compile Cython: `cd RecSys2019_DeepLearning_Evaluation`, `/opt/miniconda3/envs/user_intercon_env/bin/python run_compile_all_cython.py`

## Execution

To execute use the command `/opt/miniconda3/envs/user_intercon_env/bin/python main.py --flagfile=flagfile.cfg`.

Specify all relevant parameters on `flagfile.cfg`. The flags to be specified are:
|       Flag Name        |   Description   | Possible Values |
| ---------------------- | ---------------------------------------------------- | --------------- |
|        dataset         |                  the dataset to be used              | 'MovieLens_100k'|
|       algorithm        |  the algorithm to be used for tuning, training, etc. | ['UserKNNCFRecommender', 'MatrixFactorization_FunkSVD_Cython'] |
|       operation        |                  the operation to execute            | ['get_dataset_statistics','generate_trainset_testset', 'algorithm_hyperparameter_tuning'] |
|   ratio_split_train    |           the desired train-test splitting ratio     | float in [0,1] |
| ratio_split_validation |      the desired train-validation splitting ratio    | float in [0,1] |

## Data Preprocessing
Generate training and test sets and compute dataset properties. 

All relevant code is in `generate_trainset_testset.py`. 

To run, set the flag `operation` to `generate_trainset_testset` on `flagfile.cfg`.

The splitted train, test, and validation sets per random seed are in `./Data/`.

## Algorithms & Hyperparameter Tuning
The implemented algorithms and tuning source code of [Dacrema et. al](https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation) are used. 

For hyperparameter tuning, set the flag `operation` to `algorithm_hyperparameter_tuning` on `flagfile.cfg`.

## Model Training & Prediction
Code: `train_varying_factors.py`
Code: `train_varying_factors.py`

## Computation of Influence

## Ploting
Code: `check_independence_assumption.py`