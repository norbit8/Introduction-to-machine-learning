import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Tuple
from plotnine import *


def fit_linear_regression(mat_x: np.array, res_vec: np.array) -> Tuple[np.array, np.array]:
    """
    Linear Regression solver
    Parameters:
    :param mat_x: The design matrix X (np array)
    :param res_vec: Response Vector y (np array)
    Returns: Tuple of the coefficient vector and the singular values of X.
    """
    ones_vec = np.ones(mat_x.shape[1])  # vectors of ones
    mat_x = np.vstack((ones_vec, mat_x))  # adding ones to the matrix
    mat_x_t = mat_x.transpose()  # transposing after adding one
    return np.linalg.pinv(mat_x_t) @ res_vec, np.linalg.svd(mat_x_t, compute_uv=False)
