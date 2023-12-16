# Algorithmically Mediated User Relations: Exploring Data's Relationality in Recommender Systems

Repository for the [paper](https://github.com/AthinaKyriakou/algorithmically_mediated_user_relations/blob/main/RegulatableML_NeurIPS2023_data_governance_social_relations.pdf) "Algorithmically Mediated User Relations: Exploring Data’s Relationality in Recommender Systems" to be presented in the [Regulatable ML Workshop @ NeurIPS 2023](https://regulatableml.github.io).

## About the Project
Personalization services, such as recommender systems, operate on vast amounts of user-item interactions to provide personalized content. To do so, they identify patterns in the available interactions and group users based on pre-existing offline or online social relations, or algorithmically determined similarities and differences. We refer to the relations created between users based on algorithmically determined constructs as algorithmically mediated user relations. Taking as a case study collaborative filtering recommendation algorithms where users are interrelated by design, we empirically examine whether algorithmically should be taken into account in practice when quantifying the influence of users’ data on the recommendations of others.

![High Level Design](/high_level_design.png)

## Installation
To install the needed dependencies run:
- `conda env create -f requirements.yml`
- `conda activate user_intercon_env`

To use matrix factorization algorithms:
- `conda install -c anaconda cython`
- Compile Cython: `cd RecSys2019_DeepLearning_Evaluation`, `/opt/miniconda3/envs/user_intercon_env/bin/python run_compile_all_cython.py`

## Execution

To execute use the command `/opt/miniconda3/envs/user_intercon_env/bin/python main.py --flagfile=flagfile.cfg`.

Specify all relevant parameters on `flagfile.cfg`. The main flags to be specified are:
|       Flag Name        |   Description   | Possible Values |
| ---------------------- | ---------------------------------------------------- | --------------- |
|        dataset         |                  the dataset to be used              | 'MovieLens_100k'|
|       algorithm        |  the algorithm to be used for tuning, training, etc. | ['UserKNNCFRecommender', 'MatrixFactorization_FunkSVD_Cython'] |
|       operation        |                  the operation to execute            | ['get_dataset_statistics','generate_trainset_testset', 'algorithm_hyperparameter_tuning', 'compute_individual_influences', 'compute_group_influences', 'check_independence_assumption'] |

## Data Preprocessing
Generate training and test sets and compute dataset properties. 

All relevant code is in `generate_trainset_testset.py`. 

To run, set the flag `operation` to `generate_trainset_testset` on `flagfile.cfg`.

The splitted train, test, and validation sets per random seed are in `./Data/`.

## Algorithms & Hyperparameter Tuning
The implemented algorithms and tuning source code of [Dacrema et. al](https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation) are used. 

For hyperparameter tuning, set the flag `operation` to `algorithm_hyperparameter_tuning` on `flagfile.cfg`.

## Computation of Influence

### Individual Influence

Set the flag `operation` to `compute_individual_influences` on `flagfile.cfg`.

### Group Influence

Set the flag `operation` to `compute_group_influences` on `flagfile.cfg`.

## Ploting

Set the flag `operation` to `check_independence_assumption` on `flagfile.cfg`.