import numpy as np
import pandas as pd
from typing import Tuple


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
    return np.linalg.pinv(mat_x_t) @ res_vec, np.diag(np.linalg.svd(mat_x_t)[1])


def predict(x: np.array, coef_v: np.array) -> np.array:
    """
    Prediction function
    :param x: Design matrix.
    :param coef_v: Coefficient vector.
    :return: The prediction of the result vector.
    """
    return np.dot(x.transpose(), coef_v)


def mse(res_vec: np.array, prediction_vec: np.array) -> float:
    """
    Mean Square Error function.
    :param res_vec: Response vector.
    :param prediction_vec: Prediction vector.
    :return: The error.
    """
    return (1/float(res_vec.size)) * (np.linalg.norm(prediction_vec - res_vec)**2)


def load_data(path: str) -> np.array:
    """
    Loads the data into a matrix (np array).
    :param path: The path to the csv of the data.
    :return: Data design matrix.
    """
    try:
        data = pd.read_csv(path)
    except FileNotFoundError:
        print("FAILED TO FIND THE DATA LOCATION!")
        return
    except Exception:
        print("AN ERROR OCCURRED WHILE LOADING THE DATA!")
        return

    print(data)

if __name__ == "__main__":
    pass
