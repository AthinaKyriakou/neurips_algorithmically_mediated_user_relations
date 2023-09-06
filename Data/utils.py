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