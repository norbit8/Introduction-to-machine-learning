import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Tuple
from plotnine import *
import math


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
    data = data.loc[filt_non_positive & filt_condition & filt_year_built & filt_date & filt_id]  # apply filters
    data = data.drop_duplicates()  # drop duplicates
    data = categorical_features(data)  # address categorical features
    data = data.drop(['id', 'date', 'zipcode'], axis=1)  # drop the categorical columns and the id
    # data.to_csv("./lol.csv")  # save csv for myself
    return data


def categorical_features(data: np.array) -> np.array:
    """
    Addressing the categorical features with one hot encoding solution.
    :param data: The data in a form of an np array.
    :return: The processed data.
    """
    # addressing zip code (One hot encoding)
    zips = data['zipcode']
    data = pd.concat([data, pd.get_dummies(zips)], axis=1)
    # addressing dates (Cleaning + One hot encoding)
    dates = data['date']
    dates = pd.concat([dates.str.slice(0, 4), dates.str.slice(4, 6), dates.str.slice(6, 8)], axis=1)
    dates.columns = ['year', 'month', 'day']  # renaming the columns for easier access
    year, month, day = dates['year'], dates['month'], dates['day']
    data = pd.concat([data, pd.get_dummies(year)], axis=1)
    data = pd.concat([data, pd.get_dummies(month)], axis=1)
    data = pd.concat([data, pd.get_dummies(day)], axis=1)
    return data


def plot_singular_values(singular_values: iter):
    """
    Given some singular values plots the scree plot.
    :param singular_values: Singular values collection
    :return: ggplot.
    """
    y = singular_values
    y.sort()
    y = y[::-1]
    x = [index for index in range(1, len(singular_values) + 1)]
    df = DataFrame({'x': x, 'y': y})
    return ggplot(df, aes(x='x', y='y')) + geom_point(size=1) + geom_line() + \
    ggtitle("Scree plot of the singular values") + \
    labs(y="Singular value", x="Component Number")


def question_15(data):
    """
    loading the data and plotting the singular values
    :return: plot of the singular values (scree plot)
    """
    data = data.drop(['price'], axis=1)  # drop price
    data_np = data.transpose()
    ones_vec = np.ones(data_np.shape[1])  # vectors of ones
    mat_x = np.vstack((ones_vec, data_np))  # adding ones to the matrix
    mat_x_t = mat_x.transpose()  # transposing after adding one
    singulars = np.linalg.svd(mat_x_t, compute_uv=False)
    return plot_singular_values(singulars)


def split_data_train_and_test(data):
    """
    Splits the data into train and test-sets randomly, such that the size
    of the test set is 1/4 of the total data, and 3/4 of the data as training data.
    :param data: Not splitted data.
    :return: Splitted data.
    """
    total_data = len(data)
    np.random.seed(7)
    msk = np.random.rand(total_data) < 0.75
    train = data[msk]
    test = data[~msk]
    return train, test


def question_16(data):
    training_data, testing_data = split_data_train_and_test(data)
    real_price_vec = testing_data['price']
    testing_data = testing_data.drop(['price'], axis=1)
    testing_data = testing_data.transpose()
    ones_vec = np.ones(testing_data.shape[1])  # vectors of ones
    testing_data = np.vstack((ones_vec, testing_data))  # adding ones to the matrix
    price_vector = training_data['price']
    training_data = training_data.drop(['price'], axis=1)
    mses = []
    for i in range(1, 101):
        train_number = i / 100
        rows = math.floor(train_number*len(training_data))
        mat_x = training_data[:math.floor(train_number*len(training_data))]
        mat_x = mat_x.transpose()
        w, singulars = fit_linear_regression(mat_x, price_vector[:rows])
        pred = predict(testing_data, w)
        mses.append(mse(real_price_vec, pred))
    return mses


def plot_results(res):
    """
    plots the MSE over the test set as a function of p%
    :param res: results.
    :return: plot
    """
    x = [index for index in range(1, 101)]
    df = DataFrame({'x': x, 'y': res})
    return ggplot(df, aes(x='x', y='y')) + geom_point(size=1) + geom_line() + \
    ggtitle("MSE over the test set as a function of p%") + \
    labs(y="MSE", x="p% (precent of the data trained)")


def plot_scatter_features_values(vector_1, res_v, name):
    """
    plots the non categorical features to the screen.
    :param vector_1: the vector of the feature.
    :param res_v: the price vector.
    :param name: the name of the feature.
    :return: a plot.
    """
    cov_mat = np.cov(vector_1, res_v, ddof=1)
    sigma1 = np.std(vector_1, ddof=1)
    sigma2 = np.std(res_v, ddof=1)
    pearson_correlation = (cov_mat[1][0]) / (sigma1 * sigma2)
    df = DataFrame({'x': res_v, 'y': vector_1})
    return ggplot(df, aes(x='x', y='y')) + geom_point(size=1)+ theme_bw() + geom_line() + \
    ggtitle("Non-categorical feature ("+name+") vs the price\n The Pearson correlation is " +
            str(pearson_correlation) + "\n") + \
    labs(y=name+" feature", x="Response vector (=price)")


def feature_evaluation(mat_x: DataFrame, res_v: np.array):
    """
    feature ecaluation, creates plots for each feature vs the price.
    :param mat_x: the matrix of features
    :param res_v: the prices vector
    """
    relevant_columns = mat_x.iloc[:, :17]
    for col in range(17):
        vector_1 = relevant_columns.iloc[:, col]
        print(plot_scatter_features_values(vector_1, res_v, vector_1.name))


if __name__ == "__main__":
    PATH_TO_CSV = "kc_house_data.csv"
    data = load_data(PATH_TO_CSV)
    print(question_15(data))  # Question 15
    res = question_16(data)
    print(plot_results(res))
    res_v = data['price']
    mat_x = data.drop(['price'], axis=1)  # drop prices
    feature_evaluation(mat_x, res_v)
