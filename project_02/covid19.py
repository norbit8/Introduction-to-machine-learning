import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Tuple
from plotnine import *
import math

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
    return data


def fit_linear_regression(mat_x: np.array, res_vec: np.array) -> Tuple[np.array, np.array]:
    """
    Linear Regression solver
    Parameters:
    :param mat_x: The design matrix X (np array)
    :param res_vec: Response Vector y (np array)
    Returns: Tuple of the coefficient vector and the singular values of X.
    """
    ones_vec = np.ones(mat_x.shape[0])  # vectors of ones
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


def plot_log_detected(day_num, log_detected, prediction):
    """
    plot for the detection values Q21
    :param day_num: number of days
    :param log_detected: detected patients so far
    :param prediction: the prediction of the detected patients.
    :return: a plot.
    """
    df = DataFrame({'x': day_num,
                    'b': log_detected.to_numpy().reshape((1, len(log_detected)))[0],
                    'c': list(prediction.reshape((1, len(prediction)))[0])
                    })
    return ggplot(aes(x='x', y='b'), data=df) + geom_point(size=1) +\
           geom_line(aes(y='b'), color='black') + \
           geom_line(aes(y='c'), color='blue') +  \
           ggtitle("Log of covid19 sick people detected vs days") + \
           labs(y="Detected sick people", x="Day number")


def plot_detected(day_num, detected, prediction):
    """
    plot for the detection values Q21
    :param day_num: number of days
    :param detected: detected patients so far
    :param prediction: the prediction of the detected patients.
    :return: a plot.
    """
    df = DataFrame({'x': day_num,
                    'y': detected,
                    'c': list(map(lambda x: math.e**x, prediction.reshape((1, len(prediction)))[0]))
                    })
    return ggplot(df, aes(x='x', y='y')) + geom_point(size=1) + geom_line() + \
           geom_line(aes(y='c'), color='blue') + \
           ggtitle("covid19 sick people detected vs days") + \
           labs(y="Detected sick people", x="Day number") + scale_x_continuous()


if __name__ == "__main__":
    PATH_TO_CSV = "covid19_israel.csv"
    data = load_data(PATH_TO_CSV)
    detected = data['detected'].to_numpy()
    log_detected = DataFrame({'log_detected': [math.log(x, math.e) for x in data['detected']]})
    day_num = data['day_num'].to_numpy()
    w, singulars = fit_linear_regression(day_num.transpose(), log_detected)
    ones_vec = np.ones(day_num.shape[0])  # vectors of ones
    new_day_num = np.vstack((ones_vec, day_num))  # adding ones to the matrix
    prediction = predict(new_day_num, w)
    print(plot_log_detected(day_num, log_detected, prediction))
    print(plot_detected(day_num, detected, prediction))
