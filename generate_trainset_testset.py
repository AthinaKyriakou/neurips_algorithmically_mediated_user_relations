# Generate training and test sets
# To execute /opt/miniconda3/envs/user_intercon_env/bin/python /Users/athina/Desktop/Research/Experiments/User_Interconnectedness/generate_trainset_testset.py

import random, os, pickle
import numpy as np
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_stratified_split
import sys
sys.path.append('./RecSys2019_DeepLearning_Evaluation/')

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

MOVIELENS_DATA_SIZE = '100k' # '100k', '1M'
COL_USER = 'UserId'
COL_ITEM = 'MovieId'
COL_RATING = 'Rating'
COL_PREDICTION = 'Prediction'
COL_TIMESTAMP = 'Timestamp'
COL_NUM_RATINGS = 'num_ratings'
COL_REC = 'recommendations'
RATIO_SPLIT = 0.8

RANDOM_SEEDS = [42, 17, 36]
if MOVIELENS_DATA_SIZE == '100k':
    FP = '/Users/athina/Desktop/Research/Experiments/User_Interconnectedness/Data/MovieLens_100k/trainsets_' + str(RATIO_SPLIT)
elif MOVIELENS_DATA_SIZE == '1M':
    FP = '/Users/athina/Desktop/Research/Experiments/User_Interconnectedness/Data/MovieLens_1M/trainsets_' + str(RATIO_SPLIT)

data_df = movielens.load_pandas_df(size=MOVIELENS_DATA_SIZE, header=[COL_USER, COL_ITEM, COL_RATING])
user_Ids_arr = np.sort(data_df[COL_USER].unique()) # numpy.int64
num_users = len(user_Ids_arr)
item_Ids_arr = np.sort(data_df[COL_ITEM].unique()) # numpy.int64
num_items = len(item_Ids_arr)
print(num_users, num_items)
print('\n')

# create row and column mappers to be used for the creation of sparse matrices of train, validation, and test sets
row_mapper_dict = {}
for u in user_Ids_arr:
    row_mapper_dict[u] = u
row_mapper_path = FP + '/row_mapper_dict.pkl'
with open(row_mapper_path, 'wb') as fp:
    pickle.dump(row_mapper_dict, fp)

col_mapper_dict = {}
for i in item_Ids_arr:
    col_mapper_dict[i] = i
col_mapper_path = FP + '/col_mapper_dict.pkl'
with open(col_mapper_path, 'wb') as fp:
    pickle.dump(col_mapper_path, fp)

for r in RANDOM_SEEDS:
    print("\nr=",r)
    random.seed = r
    np.random.seed(r)
    FP_RS = FP + '/set_' + str(r)
    if not os.path.exists(FP_RS):
        os.makedirs(FP_RS)
    
    train_df, test_df = python_stratified_split(data_df, filter_by='user', ratio=RATIO_SPLIT, col_user=COL_USER, col_item=COL_ITEM, seed=r)
    #train_df, validation_df = python_stratified_split(train_df, filter_by='user', ratio=RATIO_SPLIT, col_user=COL_USER, col_item=COL_ITEM, seed=r)

    user_Ids_train_arr = np.sort(train_df[COL_USER].unique()) # ints
    num_users_train = len(user_Ids_train_arr)
    item_Ids_train_arr = np.sort(train_df[COL_ITEM].unique()) # ints
    num_items_train = len(item_Ids_train_arr)
    print("trainset:", num_users_train, num_items_train)

    user_Ids_test_arr = np.sort(test_df[COL_USER].unique()) # ints
    num_users_test = len(user_Ids_test_arr)
    item_Ids_test_arr = np.sort(test_df[COL_ITEM].unique()) # ints
    num_items_test = len(item_Ids_test_arr)
    print("testset:", num_users_test, num_items_test)

    train_path = FP_RS + '/trainset.csv'
    train_df.to_csv(train_path, index=False)
    #validation_path = FP_RS + '/validation.csv'
    #validation_df.to_csv(validation_path, index=False)
    test_path = FP_RS + '/testset.csv'
    test_df.to_csv(test_path, index=False)