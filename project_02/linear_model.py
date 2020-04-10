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
    # filt1 = data['bedrooms'] < 33
    filt_non_positive = (data['price'] > 0) & (data['sqft_lot15'] > 0) & (data['sqft_living'] > 0) & \
                        (data['floors'] > 0)
    filt_condition = (data['condition'] <= 5) & (data['condition'] >= 1)
    filt_year_built = data['yr_built'] <= 2015
    filt_date = data['date'].notnull()
    filt_id = data['id'].notnull()
    data = data.loc[filt_non_positive & filt_condition & filt_year_built & filt_date & filt_id]
    data = data.drop_duplicates()
    return data


def plot_singular_values(singular_values: iter):
    """
    Given some singular values plots the scree plot.
    :param singular_values: Singular values collection
    :return: ggplot.
    """
    y = singular_values
    y.sort(reverse=True)
    x = [index for index in range(1, len(singular_values) + 1)]
    df = DataFrame({'x': x, 'y': y})
    return ggplot(df, aes(x='x', y='y')) + geom_point(size=3) + geom_line() + \
    ggtitle("Scree plot of the singular values") + \
    labs(y="Singular value", x="Component Number")

def question_15():
    PATH_TO_CSV = "kc_house_data.csv"
    data = load_data(PATH_TO_CSV)
    data_np = data.to_numpy().transpose()
    ones_vec = np.ones(data_np.shape[1])  # vectors of ones
    mat_x = np.vstack((ones_vec, data_np))  # adding ones to the matrix
    mat_x_t = mat_x.transpose()  # transposing after adding one
    asdfasdf = np.array(mat_x, dtype='float')
    singulars = np.linalg.svd(asdfasdf, compute_uv=False)
    plot_singular_values(singulars)


if __name__ == "__main__":
    question_15()