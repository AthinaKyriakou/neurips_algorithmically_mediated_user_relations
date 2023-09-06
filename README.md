# Installation
- `conda env create -f requirements.yml`
- `conda activate user_intercon_env`

# Operations

## Generate training and test sets
File: `generate_trainset_testset.py`

## Hyperparameter tuning
File: `algorithm_hyperparameter_tuning.py`
Tuning based on Dacrema. 
To use the Matrix Factorization, need to compile Cython --> cd RecSys2019_DeepLearning_Evaluation, /opt/miniconda3/envs/user_intercon_env/bin/python run_compile_all_cython.py
to do so: conda install -c anaconda cython