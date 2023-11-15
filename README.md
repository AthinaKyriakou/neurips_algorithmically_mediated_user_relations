# Algorithmically Mediated User Relations: Exploring Data's Relationality in Recommender Systems

## Installation
To install the needed dependencies run:
- `conda env create -f requirements.yml`
- `conda activate user_intercon_env`

## Operations

### Data Preprocessing
Generate training, test and validation sets and compute dataset properties.
Code: `generate_trainset_testset.py`

MovieLens 100k and 1M are downloaded from: XX

Splitted train, test, and validation are in `./Data/`.

### Hyperparameter Tuning
Code: `algorithm_hyperparameter_tuning.py`
Tuning based on Dacrema. 
To use the Matrix Factorization, need to compile Cython --> cd RecSys2019_DeepLearning_Evaluation, /opt/miniconda3/envs/user_intercon_env/bin/python run_compile_all_cython.py
to do so: conda install -c anaconda cython

### Model Training & Prediction
Code: `train_varying_factors.py`
Code: `train_varying_factors.py`

### Computation of Influence

### Ploting
Code: `check_independence_assumption.py`