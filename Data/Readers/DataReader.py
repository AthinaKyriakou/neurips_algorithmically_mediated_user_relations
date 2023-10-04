import os
import numpy as np
from Data.Dataset import Dataset

# utility functions
from RecSys2019_DeepLearning_Evaluation.Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix
from RecSys2019_DeepLearning_Evaluation.Data_manager.load_and_save_data import save_data_dict_zip, load_data_dict_zip
from RecSys2019_DeepLearning_Evaluation.Base.Recommender_utils import reshapeSparse
#from Data_manager.split_functions.split_train_validation import split_train_validation_leave_one_out_user_wise

# adapted from recsys2019_deeplearning_evaluation/Conferences/WWW/NeuMF_our_interface/Movielens1M/Movielens1MReader.py

class DataReader(object):

    URM_DICT = {}
    ICM_DICT = {}

    def __init__(self, num_users, num_items, isHeader, trainOriginalPath, trainPath, validationPath, testPath):

        print("In DataReader...")

        super(DataReader, self).__init__() # is this used?

        try:

            print("DataReader: Attempting to load pre-splitted data")
            
            # Ensure file is loaded as matrix - why do they do it like this here??
            #Dataset.load_rating_file_as_list = Dataset.load_rating_file_as_matrix

            dataset = Dataset(num_users=num_users, num_items=num_items, isHeader=isHeader, trainOriginalPath=trainOriginalPath, trainPath=trainPath, validationPath=validationPath, testPath=testPath)
            self.mapping_matrixPos_userIds_dict = dataset.mapping_matrixPos_userIds_dict
            self.mapping_userIds_matrixPos_dict = dataset.mapping_userIds_matrixPos_dict
            print("DataReader: Dataset loaded")

            # constuct the sparse matrices: URM_train_original, URM_train, URM_validation, URM_test
            URM_train_original = dataset.trainOriginalMatrix
            URM_train = dataset.trainMatrix
            URM_validation = dataset.validationMatrix
            URM_test = dataset.testRatings
            
            URM_train_original = URM_train_original.tocsr()
            URM_train = URM_train.tocsr()
            URM_validation = URM_validation.tocsr()
            URM_test = URM_test.tocsr()

            shape = (max(URM_train_original.shape[0], URM_test.shape[0]),
                     max(URM_train_original.shape[1], URM_test.shape[1]))

            URM_train_original = reshapeSparse(URM_train_original, shape)
            URM_train = reshapeSparse(URM_train, shape)
            URM_validation = reshapeSparse(URM_validation, shape)
            URM_test = reshapeSparse(URM_test, shape)

            # construct URM_test_Negative
            # adapted from RecSys2019_DeepLearning_Evaluation/Data_manager/split_functions/split_data_on_timestamp.py
            negative_items_per_positive = 99
            URM_negative_builder = IncrementalSparseMatrix(n_rows=num_users, n_cols=num_items)
            all_items = np.arange(0, num_items, dtype=np.int)
            URM_all = URM_train_original + URM_test

            for user_index in range(URM_all.shape[0]):
                if user_index % 10000 == 0:
                    print("split_data_on_sequence: user {} of {}".format(user_index, URM_all.shape[0]))
                start_pos = URM_all.indptr[user_index]
                end_pos = URM_all.indptr[user_index+1]
                user_profile = URM_all.indices[start_pos:end_pos]
                unobserved_index = np.in1d(all_items, user_profile, assume_unique=True, invert=True)
                unobserved_items = all_items[unobserved_index]
                np.random.shuffle(unobserved_items)
                URM_negative_builder.add_single_row(user_index, unobserved_items[:negative_items_per_positive], 1.0)
            URM_negative = URM_negative_builder.get_SparseMatrix()

            # save the constructed matrices to the dict
            self.URM_DICT = {
                "URM_train_original": URM_train_original,
                "URM_train": URM_train,
                "URM_validation": URM_validation,
                "URM_test": URM_test,
                "URM_test_negative": URM_negative,
            }

        except FileNotFoundError as e:
            print(e)