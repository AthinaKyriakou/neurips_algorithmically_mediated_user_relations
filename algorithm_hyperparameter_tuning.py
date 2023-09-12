# /opt/miniconda3/envs/user_intercon_env/bin/python /Users/athina/Desktop/Research/Experiments/User_Interconnectedness/algorithm_hyperparameter_tuning.py
import pandas as pd
import numpy as np
from functools import partial
import sys, traceback, os
sys.path.append('./RecSys2019_DeepLearning_Evaluation/')

# readers
from Data.Readers.MovieLens100kReader import MovieLens100kReader

# data checks
from RecSys2019_DeepLearning_Evaluation.Utils.assertions_on_data_for_experiments import assert_disjoint_matrices
from Data.utils import assert_correct_size_df_to_sparse

# recommender algorithms
from RecSys2019_DeepLearning_Evaluation.Recommender_import_list import *
from RecSys2019_DeepLearning_Evaluation.Base.Evaluation.Evaluator import EvaluatorNegativeItemSample, EvaluatorHoldout
# utility function to create sparse user-item interaction matrix out of the training and test datasets
from RecSys2019_DeepLearning_Evaluation.Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs, IncrementalSparseMatrix
from RecSys2019_DeepLearning_Evaluation.Data_manager.split_functions.split_train_validation import split_train_validation_leave_one_out_user_wise
# utility function for hyperparameter tuning
from RecSys2019_DeepLearning_Evaluation.ParameterTuning.run_parameter_search import runParameterSearch_Collaborative

dataset_name = 'MovieLens_100k'
MOVIELENS_DATA_SIZE = '100k' # '100k', '1M'
NUM_USERS = 943
NUM_ITEMS = 1682
COL_USER = 'UserId'
COL_ITEM = 'MovieId'
COL_RATING = 'Rating'
RATIO_SPLIT = 0.8

# BASED ON: RecSys2019_DeepLearning_Evaluation/run_WWW_17_NeuMF.py

# 1. Read training and test set from file - I need to have a dataset structure
if dataset_name == 'MovieLens_100k':
    dataPath = '/Users/athina/Desktop/Research/Experiments/User_Interconnectedness/Data/MovieLens_100k/trainsets_0.8/set_42/'
    trainPath = dataPath + 'trainset.csv'
    testPath = dataPath + 'testset.csv'
    isHeader = True
    dataset = MovieLens100kReader(NUM_USERS, NUM_ITEMS, isHeader, trainPath, testPath)

# 2. Check if the sparse matrices of training and test set are properly created
URM_train = dataset.URM_DICT["URM_train"].copy()
URM_test = dataset.URM_DICT["URM_test"].copy()
assert_disjoint_matrices([URM_train, URM_test])
assert_correct_size_df_to_sparse(trainPath, URM_train.nnz)
assert_correct_size_df_to_sparse(testPath, URM_test.nnz)

# construction of a validation set - might need to change OK! check for multiple validation sets
URM_train, URM_validation = split_train_validation_leave_one_out_user_wise(URM_train.copy())
assert_disjoint_matrices([URM_train, URM_validation, URM_test])

# 3. Tuning
result_folder_path = "/Users/athina/Desktop/Research/Experiments/User_Interconnectedness/result_experiments/MovieLens_100k/trainsets_0.8/set_42/"
if not os.path.exists(result_folder_path):
    os.makedirs(result_folder_path)

collaborative_algorithm_list = [
    #UserKNNCFRecommender,
    #ItemKNNCFRecommender,
    RP3betaRecommender,
    #EASE_R_Recommender,
    #PureSVDRecommender,
    #MatrixFactorization_FunkSVD_Cython,
    ]
metric_to_optimize = "NDCG" # might need to add more

# what are these?
n_cases = 50
n_random_starts = 15

# need to create validation and test_negative sets (??): validation set 10% on every fold (how to implement it?)
# HERE: trying to find how to create URM_validation, URM_test_negative

# Approach 1: check if I need the negative items - DMF
#cutoff_list_validation = [10]
#cutoff_list_test = [5, 10, 20]
#evaluator_validation = EvaluatorNegativeItemSample(URM_validation, URM_test_negative, cutoff_list=cutoff_list_validation)
#evaluator_test = EvaluatorNegativeItemSample(URM_test, URM_test_negative, cutoff_list=cutoff_list_test)

# Approach 2: NeuRec

max_cutoff = URM_train.shape[1]-1
cutoff_list_validation = [10]
cutoff_list_test=[5, 10, 50, max_cutoff]
evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=cutoff_list_validation)
evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list_test)


runParameterSearch_Collaborative_partial = partial(runParameterSearch_Collaborative,
                                                    URM_train = URM_train,
                                                    URM_train_last_test = URM_train + URM_validation,
                                                    metric_to_optimize = metric_to_optimize,
                                                    evaluator_validation_earlystopping = evaluator_validation,
                                                    evaluator_validation = evaluator_validation,
                                                    evaluator_test = evaluator_test,
                                                    output_folder_path = result_folder_path,
                                                    parallelizeKNN = False,
                                                    allow_weighting = True,
                                                    resume_from_saved = True,
                                                    n_cases = n_cases,
                                                    n_random_starts = n_random_starts)
        
for recommender_class in collaborative_algorithm_list:
    try:
        runParameterSearch_Collaborative_partial(recommender_class)
    except Exception as e:
        print("On recommender {} Exception {}".format(recommender_class, str(e)))
        traceback.print_exc()