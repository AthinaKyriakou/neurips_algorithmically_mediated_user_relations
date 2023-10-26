# Generate training and test sets
# To execute /opt/miniconda3/envs/user_intercon_env/bin/python /Users/athina/Desktop/Research/Experiments/User_Interconnectedness/1_generate_trainset_testset.py

import random, os, pickle, math, json
import numpy as np
import pandas as pd
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_stratified_split, python_random_split
from cornac.datasets import citeulike
from Data.utils import df_space, df_shape, df_density, df_gini_user, df_gini_item
import sys
sys.path.append('./RecSys2019_DeepLearning_Evaluation/')

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

DATASET = 'CiteULike' # 'MovieLens_100k', 'MovieLens_1M', 'Amazon_Movies_and_TV', 'CiteULike', 'GoodReads_Comics', 'Pinterest'
COL_USER = 'UserId'
COL_ITEM = 'ItemId'
COL_RATING = 'Rating'
COL_INTERACTION = 'Interaction'
COL_PREDICTION = 'Prediction'
COL_TIMESTAMP = 'Timestamp'
COL_NUM_RATINGS = 'num_ratings'
COL_REC = 'recommendations'
RATIO_SPLIT_TRAIN = 0.8
RATIO_SPLIT_VALIDATION = 0.9
RANDOM_SEEDS = [42, 17, 36]

if DATASET == 'MovieLens_100k':
    FP = '/Users/athina/Desktop/Research/Experiments/User_Interconnectedness/Data/MovieLens_100k/trainsets_' + str(RATIO_SPLIT_TRAIN)
    data_df = movielens.load_pandas_df(size='100k', header=[COL_USER, COL_ITEM, COL_RATING])
elif DATASET == 'MovieLens_1M':
    FP = '/Users/athina/Desktop/Research/Experiments/User_Interconnectedness/Data/MovieLens_1M/trainsets_' + str(RATIO_SPLIT_TRAIN)
    data_df = movielens.load_pandas_df(size='1M', header=[COL_USER, COL_ITEM, COL_RATING])
elif DATASET == 'Amazon_Movies_and_TV':
    #URL: https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/Movies_and_TV.csv
    FP = '/Users/athina/Desktop/Research/Experiments/User_Interconnectedness/Data/Amazon_Movies_and_TV/'
    '''
    # Replace user and product Ids to integers
    # itemId, userId, rating, timestamp
    dataPath = FP + 'Movies_and_TV.csv'
    data_df = pd.read_csv(dataPath, names=[COL_ITEM, COL_USER, COL_RATING, COL_TIMESTAMP]) 
    # map user ids to integers
    unique_users = data_df[COL_USER].unique()
    user_map = {u: i for i, u in enumerate(unique_users)}
    data_df[COL_USER] = data_df[COL_USER].map(user_map)
    # map product ids to integers  
    unique_items = data_df[COL_ITEM].unique()
    item_map = {i: j for j, i in enumerate(unique_items)}
    data_df[COL_ITEM] = data_df[COL_ITEM].map(item_map)
    # write updated file to csv
    dataPath = FP + 'Movies_and_TV_updated.csv'
    data_df.to_csv(dataPath, header=False, index=False)
    '''
    '''
    # Remove all users with less than 20 interactions
    # itemId, userId, rating, timestamp
    dataPath = FP + 'Movies_and_TV.csv'
    data_df = pd.read_csv(dataPath, names=[COL_ITEM, COL_USER, COL_RATING, COL_TIMESTAMP])
    data_df = data_df[data_df[COL_USER].map(data_df[COL_USER].value_counts()) > 20] 
    # map user ids to integers
    unique_users = data_df[COL_USER].unique()
    user_map = {u: i for i, u in enumerate(unique_users)}
    data_df[COL_USER] = data_df[COL_USER].map(user_map)
    # map product ids to integers  
    unique_items = data_df[COL_ITEM].unique()
    item_map = {i: j for j, i in enumerate(unique_items)}
    data_df[COL_ITEM] = data_df[COL_ITEM].map(item_map)
    # increase user and item ids by 1
    data_df[COL_USER] += 1
    data_df[COL_ITEM] += 1 
    # write updated file to csv
    dataPath = FP + 'Movies_and_TV_preprocessed_updated.csv'
    new_data_df = data_df[[COL_USER, COL_ITEM, COL_RATING]].copy()
    new_data_df = new_data_df.drop_duplicates()
    new_data_df.to_csv(dataPath, header=False, index=False)
    sys.exit()
    '''
    dataPath = FP + 'Movies_and_TV_preprocessed_updated.csv'
    data_df = pd.read_csv(dataPath, names=[COL_USER, COL_ITEM, COL_RATING]) 
    FP = FP + 'trainsets_' + str(RATIO_SPLIT_TRAIN)
elif DATASET == 'CiteULike':
    #URL: https://cornac.readthedocs.io/en/latest/datasets.html#module-cornac.datasets.citeulike
    FP = '/Users/athina/Desktop/Research/Experiments/User_Interconnectedness/Data/CiteULike/'
    '''
    data = citeulike.load_feedback() # list of tuples (user, item, 1)
    data_df = pd.DataFrame(data, columns =[COL_USER, COL_ITEM, COL_INTERACTION])
    data_df = data_df[data_df[COL_USER].map(data_df[COL_USER].value_counts()) > 20] 
    # map user and item ids to integers
    unique_users = data_df[COL_USER].unique()
    user_map = {u: i for i, u in enumerate(unique_users)}
    data_df[COL_USER] = data_df[COL_USER].map(user_map)
    unique_items = data_df[COL_ITEM].unique()
    item_map = {i: j for j, i in enumerate(unique_items)}
    data_df[COL_ITEM] = data_df[COL_ITEM].map(item_map)
    data_df[COL_USER] += 1
    data_df[COL_ITEM] += 1
    new_data_df = data_df[[COL_USER, COL_ITEM, COL_INTERACTION]].copy().drop_duplicates()
    dataPath = FP + 'CiteULike_updated.csv'
    new_data_df.to_csv(dataPath, header=False, index=False)
    sys.exit()
    '''
    dataPath = FP + 'CiteULike_updated.csv'
    data_df = pd.read_csv(dataPath, names=[COL_USER, COL_ITEM, COL_INTERACTION])
    FP = FP + 'trainsets_' + str(RATIO_SPLIT_TRAIN)
elif DATASET == 'Pinterest':
    FP = '/Users/athina/Desktop/Research/Experiments/User_Interconnectedness/Data/Pinterest/'
    '''
    # According to https://dl.acm.org/doi/10.1145/3488560.3498519 to keep users with at least 20 interactions
    # https://github.com/almightyGOSU/TheDatasetsDilemma/blob/cf095f22c0795f6c15001534461f9ac629bbb0cc/Step%202/util.py#L130
    dataPath = FP + 'pinterest.csv'
    data_df = pd.read_csv(dataPath, names=[COL_USER, COL_ITEM, COL_INTERACTION])
    data_df = data_df[data_df[COL_USER].map(data_df[COL_USER].value_counts()) > 20] 
    # write updated file to csv
    dataPath = FP + 'pinterest_preprocessed.csv'
    new_data_df = data_df[[COL_USER, COL_ITEM, COL_INTERACTION]].copy()
    new_data_df = new_data_df.drop_duplicates()
    new_data_df.to_csv(dataPath, header=False, index=False)
    sys.exit()
    '''
    dataPath = FP + 'pinterest_preprocessed.csv'
    data_df = pd.read_csv(dataPath, names=[COL_USER, COL_ITEM, COL_INTERACTION])
    FP = FP + 'trainsets_' + str(RATIO_SPLIT_TRAIN)
elif DATASET == 'GoodReads_Comics':
    FP = '/Users/athina/Desktop/Research/Experiments/User_Interconnectedness/Data/Lastfm_360/trainsets_' + str(RATIO_SPLIT_TRAIN)
    dataPath = '/Users/athina/Desktop/Research/Experiments/User_Interconnectedness/Data/GoodReads_Comics/goodreads_interactions_comics_graphic.json'
    with open(dataPath, "r") as f:
        data = json.load(f)
        print(type(data))
        sys.exit()
    print("todo")
else:
    print("Dataset not considered yet!")
    sys.exit()

# Get dataset properties: log space, log shape, log density, gini user, gini item
num_users = max(data_df[COL_USER]) # numpy.int64
num_items = max(data_df[COL_ITEM]) # numpy.int64
print("\nStatistics for:", DATASET)
print("#Users: ", num_users)
print("#Items: ", num_items)
print("#Interactions: ", data_df.shape[0])
print("spaceSizeLog: ", round(math.log10(df_space(data_df, COL_USER, COL_ITEM)), 3))
print("shapeLog: ", round(math.log10(df_shape(data_df, COL_USER, COL_ITEM)), 3))
print("density: ", round(df_density(data_df, COL_USER, COL_ITEM), 3))
print("densityLog: ", round(math.log10(df_density(data_df, COL_USER, COL_ITEM)), 3))
print("userGini: ", round(df_gini_user(data_df, COL_USER), 3))
print("itemGini: ", round(df_gini_item(data_df, COL_ITEM), 3))

'''
# create train, test, and validation sets for each random seed 
for r in RANDOM_SEEDS:
    print("\nr =",r)
    random.seed = r
    np.random.seed(r)
    FP_RS = FP + '/set_' + str(r)
    if not os.path.exists(FP_RS):
        os.makedirs(FP_RS)
    
    train_original_df, test_df = python_stratified_split(data_df, filter_by='user', ratio=RATIO_SPLIT_TRAIN, col_user=COL_USER, col_item=COL_ITEM, seed=r)
    train_df, validation_df = python_random_split(train_original_df, ratio=RATIO_SPLIT_VALIDATION, seed=r)

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

    train_original_path = FP_RS + '/trainset_original.csv'
    train_original_df.to_csv(train_original_path, index=False)
    train_path = FP_RS + '/trainset.csv'
    train_df.to_csv(train_path, index=False)
    validation_path = FP_RS + '/validation.csv'
    validation_df.to_csv(validation_path, index=False)
    test_path = FP_RS + '/testset.csv'
    test_df.to_csv(test_path, index=False)
'''