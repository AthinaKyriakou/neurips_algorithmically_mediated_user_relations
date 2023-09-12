import numpy as np
import pandas as pd

def assert_correct_size_df_to_sparse(csvPath, sparseMatrixSize):
    """
    Checks whether the number of user-item interactions in the dataframe is the same as in the sparse matrix
    :param csvPath: path of the csv file with user-item interactions
    :param sparseMatrixSize: size of the created out of the dataframe matrix
    :return:
    """
    df = pd.read_csv(csvPath)

    assert df.shape[0] == sparseMatrixSize, \
        "assert_correct_size_df_to_sparse: df and sparse matrix do not have the same number of user-item interactions"

    print("Assertion assert_correct_size_df_to_sparse: Passed")

# Utils for dataset metrics
# Gediminas Adomavicius and Jingjing Zhang. 2012. Impact of data characteristics on recommender systems performance. ACM Trans. Manage. Inf. Syst. 3, 1, Article 3 (April 2012), 17 pages. https://doi.org/10.1145/2151163.2151166
# Jin Yao Chin, Yile Chen, and Gao Cong. 2022. The Datasets Dilemma: How Much Do We Really Know About Recommendation Datasets? In Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining (WSDM '22). Association for Computing Machinery, New York, NY, USA, 141–149. https://doi.org/10.1145/3488560.3498519
def df_space(df, col_user, col_item):
    """
    Computes the space of a dataset of user-item interactions
    :param df: dataset's dataframe
    :param col_user: name of the column corresponding to user ids
    :param col_item: name of the column corresponding to user ids
    :return: space
    """
    num_users = len(df[col_user].unique())
    num_items = len(df[col_item].unique())
    return num_users * num_items

def df_shape(df, col_user, col_item):
    """
    Computes the shape of a dataset of user-item interactions
    :param df: dataset's dataframe
    :param col_user: name of the column corresponding to user ids
    :param col_item: name of the column corresponding to user ids
    :return: shape
    """
    num_users = len(df[col_user].unique())
    num_items = len(df[col_item].unique())
    return num_users / num_items

def df_density(df, col_user, col_item):
    """
    Computes the density of a dataset of user-item interactions
    :param df: dataset's dataframe
    :param col_user: name of the column corresponding to user ids
    :param col_item: name of the column corresponding to user ids
    :return: density
    """
    num_users = len(df[col_user].unique())
    num_items = len(df[col_item].unique())
    num_ratings = df.shape[0]
    return num_ratings / (num_users * num_items)

def df_gini_user(df, col_user):
    """
    Computes the user gini index in a dataset of user-item interactions
    :param df: dataset's dataframe
    :param col_user: name of the column corresponding to user ids
    :return: gini_user
    """
    users_arr = np.sort(df[col_user].unique())
    num_users = len(users_arr)
    num_ratings = df.shape[0]
    sum = 0
    for i in users_arr:
        a = (num_users + 1 - i) / (num_users + 1)
        ratings_i = df.loc[df[col_user] == i].shape[0]
        b = ratings_i / num_ratings
        sum += a * b
    return 1 - 2 * sum

def df_gini_item(df, col_item):
    """
    Computes the item gini index in a dataset of user-item interactions
    :param df: dataset's dataframe
    :param col_item: name of the column corresponding to item ids
    :return: gini_item
    """
    items_arr = np.sort(df[col_item].unique())
    num_items = len(items_arr)
    num_ratings = df.shape[0]
    sum = 0
    for i in items_arr:
        a = (num_items + 1 - i) / (num_items + 1)
        ratings_i = df.loc[df[col_item] == i].shape[0]
        b = ratings_i / num_ratings
        sum += a * b
    return 1 - 2 * sum