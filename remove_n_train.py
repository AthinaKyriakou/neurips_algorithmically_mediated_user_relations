import numpy as np
import pandas as pd
import sys, os
sys.path.append('./RecSys2019_DeepLearning_Evaluation/')
from RecSys2019_DeepLearning_Evaluation.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_FunkSVD_Cython
from RecSys2019_DeepLearning_Evaluation.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from itertools import combinations

# readers
from Data.Readers.DataReader import DataReader

# data checks
from recommenders.datasets import movielens
from RecSys2019_DeepLearning_Evaluation.Utils.assertions_on_data_for_experiments import assert_disjoint_matrices
from Data.utils import assert_correct_size_df_to_sparse

# recommender models
from RecSys2019_DeepLearning_Evaluation.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_FunkSVD_Cython
from RecSys2019_DeepLearning_Evaluation.KNN.UserKNNCFRecommender import UserKNNCFRecommender

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from absl import flags
FLAGS = flags.FLAGS

def remove_n_train(dataset, model):

    # 1. Setup file paths and load the datasets
    result_folder_path = './result_experiments/'
    if dataset == 'MovieLens_100k':
        FP = './Data/MovieLens_100k/trainsets_' + str(FLAGS.ratio_split_train) + '/set_' + str(FLAGS.random_seed)
        trainOriginalPath = FP + '/trainset_original.csv'
        trainPath = FP + '/trainset.csv'
        validationPath = FP + '/validation.csv'
        testPath = FP + '/testset.csv'
        isHeader = True

        MODELS_PATH = result_folder_path + 'MovieLens_100k/trainsets_' + str(FLAGS.ratio_split_train) + '/set_' + str(FLAGS.random_seed) + '/EvaluatorNegativeItemSample/' + model + '/experiment_influence/'
        REDUCED_TRAINSETS_PATH = FP + '/experiment_influence_reduced_datasets/'

        # load the full dataset
        print("\nfull dataset")
        data_df = movielens.load_pandas_df(size='100k', header=[FLAGS.col_user, FLAGS.col_item, FLAGS.col_rating]) #TODO: load a stored version of the dataset
        num_users = max(data_df[FLAGS.col_user])
        num_items = max(data_df[FLAGS.col_item]) 
        print("users:", num_users)
        print("items:", num_items)

        # load the training set
        print("\ntraining set")
        train_df = pd.read_csv(trainOriginalPath, header=0)
        user_Ids_arr = np.sort(train_df[FLAGS.col_user].unique()) # ints, no need to reduce by 1 since we are not making predictions
        item_Ids_arr = np.sort(train_df[FLAGS.col_item].unique()) # ints
        print("users:", len(user_Ids_arr))
        print("items:", len(item_Ids_arr))
    else:
        print("Dataset not considered yet!")
        sys.exit()
    
    if model != 'UserKNNCFRecommender' and model != 'MatrixFactorization_FunkSVD_Cython_unbiased':
        print("Model not considered yet!")
        sys.exit()


    # 2. Train and save the model without all user subsets of size 1
    for s in range(1, 2):
        print('subset length: ', s)
        subsets_uA = list(combinations(user_Ids_arr, s))

        s_path = MODELS_PATH + str(s) + '/'
        trainsets_path = REDUCED_TRAINSETS_PATH + str(s) + '/'
        print('\n', s_path)
        if not os.path.exists(s_path):
            print('Creating results folder for subsets of length s=' + str(s))
            os.makedirs(s_path)

        for subset_tuple in subsets_uA:
            print('new tuple: ', subset_tuple)
            cur_v = subset_tuple[0]
            dir_path = s_path + str(cur_v) + '/'
            print('\n',dir_path)
            if not os.path.exists(dir_path):
                print('Creating folder for ' + str(cur_v))
                os.makedirs(dir_path)                

            # remove the elements of the subset tuple from the training dataset
            new_train_df = train_df
            subset_path = ''
            for u in subset_tuple:
                new_train_df = new_train_df[new_train_df.UserId != u]
                subset_path = subset_path + str(u) + '_'
            
            # construct the dataset with the new trainset
            trainOriginalNewPath = trainsets_path + str(cur_v)
            print('\n', trainOriginalNewPath)
            if not os.path.exists(trainOriginalNewPath):
                print('Creating trainset folder for ' + str(cur_v))
                os.makedirs(trainOriginalNewPath) 
            trainOriginalNewPath = trainOriginalNewPath + '/trainset_original_new.csv'
            new_train_df.to_csv(trainOriginalNewPath, index=False)
            dataset_struct = DataReader(num_users, num_items, isHeader, trainOriginalNewPath, trainPath, validationPath, testPath)
            
            URM_new_train = dataset_struct.URM_DICT['URM_train_original'].copy()
            URM_test = dataset_struct.URM_DICT['URM_test'].copy()
            URM_test_negative = dataset_struct.URM_DICT['URM_test_negative'].copy()
            assert_disjoint_matrices([URM_new_train, URM_test, URM_test_negative])
            assert_correct_size_df_to_sparse(trainOriginalNewPath, URM_new_train.nnz)
            assert_correct_size_df_to_sparse(testPath, URM_test.nnz)

            # setup the recommender model & re-train with the new_train_df
            print("\n\n TODO: READ MODEL PARAMETERS FROM FILE")
            if dataset == 'MovieLens_100k':
                if model == 'MatrixFactorization_FunkSVD_Cython_unbiased':
                    recommender = MatrixFactorization_FunkSVD_Cython(URM_new_train)
                    recommender.fit(random_seed=FLAGS.random_seed, sgd_mode='adagrad', epochs=365, use_bias=False, batch_size=1, num_factors=195, item_reg=4.992453416923983e-05, user_reg=1.8699039697141504e-05, 
                            learning_rate=0.007164040428191017, negative_interactions_quota=0.19123099563358142)
                elif model == 'UserKNNCFRecommender':
                    recommender = UserKNNCFRecommender(URM_new_train)
                    recommender.fit(topK=185, shrink=0, similarity='cosine', normalize=True, feature_weighting = 'none')
            
            # save the model
            recommender.save_model(dir_path)
            #break
        #break