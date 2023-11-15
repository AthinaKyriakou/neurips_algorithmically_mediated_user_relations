# Generate training and test sets

import os, math
import numpy as np
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_stratified_split, python_random_split
from Data.utils import df_space, df_shape, df_density, df_gini_user, df_gini_item
import sys
sys.path.append('./RecSys2019_DeepLearning_Evaluation/')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from absl import flags
FLAGS = flags.FLAGS

def load_dataset(dataset_name):
    if dataset_name == 'MovieLens_100k':
        data_df = movielens.load_pandas_df(size='100k', header=[FLAGS.col_user, FLAGS.col_item, FLAGS.col_rating])
    elif dataset_name == 'MovieLens_1M':
        data_df = movielens.load_pandas_df(size='1M', header=[FLAGS.col_user, FLAGS.col_item, FLAGS.col_rating])
    return data_df

def get_dataset_statistics(dataset_name):
    # load the dataset
    data_df = load_dataset(dataset_name)

    # Dataset properties: log space, log shape, log density, gini user, gini item
    num_users = max(data_df[FLAGS.col_user]) # numpy.int64
    num_items = max(data_df[FLAGS.col_item]) # numpy.int64
    print("\nStatistics for:", dataset_name)
    print("#Users: ", num_users)
    print("#Items: ", num_items)
    print("#Interactions: ", data_df.shape[0])
    print("spaceSizeLog: ", round(math.log10(df_space(data_df, FLAGS.col_user, FLAGS.col_item)), 3))
    print("shapeLog: ", round(math.log10(df_shape(data_df, FLAGS.col_user, FLAGS.col_item)), 3))
    print("density: ", round(df_density(data_df, FLAGS.col_user, FLAGS.col_item), 3))
    print("densityLog: ", round(math.log10(df_density(data_df, FLAGS.col_user, FLAGS.col_item)), 3))
    print("userGini: ", round(df_gini_user(data_df, FLAGS.col_user), 3))
    print("itemGini: ", round(df_gini_item(data_df, FLAGS.col_user), 3))

def generate_trainset_testset(dataset_name, printStatistics=False):
    
    # load the dataset
    data_df = load_dataset(dataset_name)
    
    # print dataset statistics
    if printStatistics:
        get_dataset_statistics(dataset_name)

    # path setup
    dataPath = './Data/' + dataset_name + '/trainsets_' + str(FLAGS.ratio_split_train) + '/set_' + str(FLAGS.random_seed)
    if not os.path.exists(dataPath):
        os.makedirs(dataPath)

    # split
    train_original_df, test_df = python_stratified_split(data_df, filter_by='user', ratio=FLAGS.ratio_split_train, col_user=FLAGS.col_user, col_item=FLAGS.col_item, seed=FLAGS.random_seed)
    train_df, validation_df = python_random_split(train_original_df, ratio=FLAGS.ratio_split_validation, seed=FLAGS.random_seed)

    # print train and testset statistics
    user_Ids_train_arr = np.sort(train_df[FLAGS.col_user].unique()) # ints
    num_users_train = len(user_Ids_train_arr)
    item_Ids_train_arr = np.sort(train_df[FLAGS.col_item].unique()) # ints
    num_items_train = len(item_Ids_train_arr)
    print("trainset num users:", num_users_train)
    print("trainset num items:", num_items_train)

    user_Ids_test_arr = np.sort(test_df[FLAGS.col_user].unique()) # ints
    num_users_test = len(user_Ids_test_arr)
    item_Ids_test_arr = np.sort(test_df[FLAGS.col_item].unique()) # ints
    num_items_test = len(item_Ids_test_arr)
    print("testset num users:", num_users_test)
    print("testset num items:", num_items_test)

    # save results to csv
    train_original_path = dataPath + '/trainset_original.csv'
    train_original_df.to_csv(train_original_path, index=False)
    train_path = dataPath + '/trainset.csv'
    train_df.to_csv(train_path, index=False)
    validation_path = dataPath + '/validation.csv'
    validation_df.to_csv(validation_path, index=False)
    test_path = dataPath + '/testset.csv'
    test_df.to_csv(test_path, index=False)