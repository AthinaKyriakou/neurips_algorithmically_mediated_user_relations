import os

from Data.Dataset import Dataset

# utility functions
from RecSys2019_DeepLearning_Evaluation.Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix
from RecSys2019_DeepLearning_Evaluation.Data_manager.load_and_save_data import save_data_dict_zip, load_data_dict_zip
from RecSys2019_DeepLearning_Evaluation.Base.Recommender_utils import reshapeSparse
#from Data_manager.split_functions.split_train_validation import split_train_validation_leave_one_out_user_wise

# adapted from recsys2019_deeplearning_evaluation/Conferences/WWW/NeuMF_our_interface/Movielens1M/Movielens1MReader.py

class MovieLens100kReader(object):

    URM_DICT = {}
    ICM_DICT = {}

    def __init__(self, num_users, num_items, isHeader, trainPath, testPath):

        print("In MovieLens100kReader...")

        super(MovieLens100kReader, self).__init__() # is this used?

        try:

            print("MovieLens100kReader: Attempting to load pre-splitted data")
            
            # Ensure file is loaded as matrix - why do they do it like this here??
            #Dataset.load_rating_file_as_list = Dataset.load_rating_file_as_matrix

            dataset = Dataset(num_users=num_users, num_items=num_items, isHeader=isHeader, trainPath=trainPath, testPath=testPath)
            print("MovieLens100kReader: Dataset loaded")

            URM_train_original, URM_test = dataset.trainMatrix, dataset.testRatings

            URM_train_original = URM_train_original.tocsr()
            URM_test = URM_test.tocsr()

            shape = (max(URM_train_original.shape[0], URM_test.shape[0]),
                     max(URM_train_original.shape[1], URM_test.shape[1]))


            URM_train_original = reshapeSparse(URM_train_original, shape)
            URM_test = reshapeSparse(URM_test, shape)

            self.URM_DICT = {
                "URM_train_original": URM_train_original,
                "URM_train": URM_train_original, # to change if I do the validation split here
                "URM_test": URM_test,
                #"URM_test_negative": URM_test_negative,
                #"URM_validation": URM_validation,
            }

            # HERE
            #save_data_dict_zip(self.URM_DICT, self.ICM_DICT, pre_splitted_path, pre_splitted_filename)


            '''
            URM_test_negatives_builder = IncrementalSparseMatrix(n_rows=shape[0], n_cols=shape[1])

            for user_index in range(len(dataset.testNegatives)):

                user_test_items = dataset.testNegatives[user_index]

                URM_test_negatives_builder.add_single_row(user_index, user_test_items, data=1.0)


            URM_test_negative = URM_test_negatives_builder.get_SparseMatrix()


            
            URM_train, URM_validation = split_train_validation_leave_one_out_user_wise(URM_train_original.copy())



            self.URM_DICT = {
                "URM_train_original": URM_train_original,
                "URM_train": URM_train,
                "URM_test": URM_test,
                "URM_test_negative": URM_test_negative,
                "URM_validation": URM_validation,
            }

            save_data_dict_zip(self.URM_DICT, self.ICM_DICT, pre_splitted_path, pre_splitted_filename)
            '''


        except FileNotFoundError as e:
            print(e)