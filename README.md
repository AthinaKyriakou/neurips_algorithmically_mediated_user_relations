# Algorithmically Mediated User Relations: Exploring Data's Relationality in Recommender Systems

Repository for the [paper](https://github.com/AthinaKyriakou/algorithmically_mediated_user_relations/blob/main/RegulatableML_NeurIPS2023_data_governance_social_relations.pdf) "Algorithmically Mediated User Relations: Exploring Data’s Relationality in Recommender Systems" presented in the [Regulatable ML Workshop @ NeurIPS 2023](https://regulatableml.github.io).

## About the Project
Personalization services, such as recommender systems, operate on vast amounts of user-item interactions to provide personalized content. To do so, they identify patterns in the available interactions and group users based on pre-existing offline or online social relations, or algorithmically determined similarities and differences. We refer to the relations created between users based on algorithmically determined constructs as algorithmically mediated user relations. Taking as a case study collaborative filtering recommendation algorithms where users are interrelated by design, we empirically examine whether algorithmically mediated user relations should be taken into account in practice when quantifying the influence of users’ data on the recommendations of others.

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
|        dataset         |                  the dataset to be used              | 'sample-test', 'MovieLens_100k'|
|       algorithm        |  the algorithm to be used for tuning, training, etc. | ['UserKNNCFRecommender', 'MatrixFactorization_FunkSVD_Cython'] |
|       operation        |                  the operation to execute            | ['get_dataset_statistics','generate_trainset_testset', 'algorithm_hyperparameter_tuning', 'compute_individual_influences', 'compute_group_influences', 'check_independence_assumption'] |

## Data Preprocessing

All relevant code is in `data_preprocessing.py`. 

To get [common statistics](https://dl.acm.org/doi/10.1145/3488560.3498519) of a given dataset, set the flag `operation` to `get_dataset_statistics` on `flagfile.cfg`.

To generate the training, validation, and test sets of a given dataset and compute statistics of the resulting sets, set the flag `operation` to `generate_trainset_testset` on `flagfile.cfg` and specify the flags:
|       Flag Name        |             Description              | Possible Values |
| ---------------------- | ------------------------------------ | --------------- |
|   ratio_split_train    |    the train-test splitting ratio    |  float in [0,1] |
| ratio_split_validation | the train-validation splitting ratio |  float in [0,1] |

The splitted train, test, and validation sets for the specified random seed are in `./Data/`.

## Algorithms & Hyperparameter Tuning
The implemented algorithms and tuning source code of [Dacrema et. al](https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation) are used. 

For hyperparameter tuning, set the flag `operation` to `algorithm_hyperparameter_tuning` on `flagfile.cfg`.

## Computation of Influence

### Individual Influence

Set the flag `operation` to `compute_individual_influences` on `flagfile.cfg`.

Relevant code:
* `remove_n_train.py`: remove each user's data from the training set and re-train the model with the reduced training set
* `compute_ratings.py`: compute the ratings of all users for each re-trained model
* `compute_individual_influence.py`: compute the influence of each user

### Group Influence

As discussed in the paper, we define 2 types of influnce of a group of users. To compute:
* $`I_{independence}`$ set the flag `operation` to `compute_group_influences_independence` on `flagfile.cfg`. Individual influences need to be computed first. Relevant code:
    * `find_influential_users.py`:
* $`I_{relations}`$ set the flag `operation` to `compute_group_influences_relations` on `flagfile.cfg`. Individual influences need to be compute first. Relevant code:
    * `find_influential_users.py`:
    * `remove_n_train_group.py`

## Ploting

Set the flag `operation` to `check_independence_assumption` on `flagfile.cfg`.