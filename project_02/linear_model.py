import numpy as np
from typing import Tuple


def fit_linear_regression(mat_x: np.array, res_vec: np.array) -> Tuple[np.array, np.array]:
    """

    Parameters:

    Returns:

    """
    ones_vec = np.ones(mat_x.shape[1])  # vectors of ones
    mat_x = np.vstack((ones_vec, mat_x))  # adding ones to the matrix
    mat_x_t = mat_x.transpose()  # transposing after adding one
    return np.linalg.pinv(mat_x_t) @ res_vec, np.diag(np.linalg.svd(mat_x_t)[1])

def predict(x: np.array, coef_v: np.array) -> np.array:
    

if __name__ == "__main__":
    pass
