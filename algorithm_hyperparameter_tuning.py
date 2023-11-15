# Code based on RecSys2019_DeepLearning_Evaluation/run_WWW_17_NeuMF.py and RecSys2019_DeepLearning_Evaluation/run_IJCAI_17_DMF.py
import pandas as pd
import numpy as np
from functools import partial
import sys, traceback, os
sys.path.append('./RecSys2019_DeepLearning_Evaluation/')

# readers
from Data.Readers.DataReader import DataReader

# data checks
from recommenders.datasets import movielens
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

from absl import flags
FLAGS = flags.FLAGS

def algorithm_hyperparameter_tuning(dataset_name, algorithm_name):
    print("here")

    result_folder_path = './result_experiments/'
    if dataset_name == 'MovieLens_100k':
        FP = './Data/MovieLens_100k/trainsets_' + str(FLAGS.ratio_split_train) + '/set_' + str(FLAGS.random_seed)
        result_folder_path = result_folder_path + 'MovieLens_100k/trainsets_' + str(FLAGS.ratio_split_train) + '/set_' + str(FLAGS.random_seed) + '/'
        data_df = movielens.load_pandas_df(size='100k', header=[FLAGS.col_user, FLAGS.col_item, FLAGS.col_rating])
    elif dataset_name == 'MovieLens_1M':
        FP = './Data/MovieLens_1M/trainsets_' + str(FLAGS.ratio_split_train) + '/set_' + str(FLAGS.random_seed)
        result_folder_path = result_folder_path + 'MovieLens_1M/trainsets_' + str(FLAGS.ratio_split_train) + '/set_' + str(FLAGS.random_seed) + '/'
        data_df = movielens.load_pandas_df(size='1M', header=[FLAGS.col_user, FLAGS.col_item, FLAGS.col_rating])
    else:
        print("Dataset not considered yet!")
        sys.exit()

    if algorithm_name == "UserKNNCFRecommender":
        collaborative_algorithm_list = [UserKNNCFRecommender]
    elif algorithm_name == "MatrixFactorization_FunkSVD_Cython":
        # add/remove bias if needed: /RecSys2019_DeepLearning_Evaluation/ParameterTuning/run_parameter_search.py
        collaborative_algorithm_list = [ MatrixFactorization_FunkSVD_Cython]
    else:
        print("Algorithm not considered yet!")
        sys.exit()

    # number of users and number of items in the full dataset
    num_users = max(data_df[FLAGS.col_user])
    num_items = max(data_df[FLAGS.col_item])

    # 1. Read training and test set from file in a dataset structure
    trainOriginalPath = FP + '/trainset_original.csv'
    trainPath = FP + '/trainset.csv'
    validationPath = FP + '/validation.csv'
    testPath = FP + '/testset.csv'
    isHeader = True
    dataset = DataReader(num_users, num_items, isHeader, trainOriginalPath, trainPath, validationPath, testPath)

    # 2. Check if the sparse matrices of training, validation and test set are properly created
    URM_train = dataset.URM_DICT['URM_train'].copy()
    URM_validation = dataset.URM_DICT['URM_validation'].copy()
    URM_test = dataset.URM_DICT['URM_test'].copy()
    URM_test_negative = dataset.URM_DICT['URM_test_negative'].copy()
    assert_disjoint_matrices([URM_train, URM_validation, URM_test, URM_test_negative])
    assert_correct_size_df_to_sparse(trainPath, URM_train.nnz)
    assert_correct_size_df_to_sparse(validationPath, URM_validation.nnz)
    assert_correct_size_df_to_sparse(testPath, URM_test.nnz)

    # 3. Tuning
    result_folder_path = result_folder_path + 'EvaluatorNegativeItemSample/' + algorithm_name + '/'
    print(result_folder_path)
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)

    # initial
    metric_to_optimize = "NDCG"
    n_cases = 50
    n_random_starts = 15
    cutoff_list_validation = [10]
    cutoff_list_test = [5, 10, 20]

    # initial
    #evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=cutoff_list_validation)
    #evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list_test)
    #runParameterSearch_Collaborative_partial = partial(runParameterSearch_Collaborative,
    #                                                    URM_train = URM_train,
    #                                                    URM_train_last_test = URM_train + URM_validation,
    #                                                    metric_to_optimize = metric_to_optimize,
    #                                                    evaluator_validation_earlystopping = evaluator_validation,
    #                                                    evaluator_validation = evaluator_validation,
    #                                                    evaluator_test = evaluator_test,
    #                                                    output_folder_path = result_folder_path,
    #                                                    parallelizeKNN = False,
    #                                                    allow_weighting = True,
    #                                                    resume_from_saved = True,
    #                                                    n_cases = n_cases,
    #                                                    n_random_starts = n_random_starts)

    # explicit feedback tuning based on: RecSys2019_DeepLearning_Evaluation/run_IJCAI_17_DMF.py
    evaluator_validation = EvaluatorNegativeItemSample(URM_validation, URM_test_negative, cutoff_list=cutoff_list_validation)
    evaluator_test = EvaluatorNegativeItemSample(URM_test, URM_test_negative, cutoff_list=cutoff_list_test)
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