# Programmed by Yoav Levy
# ID 314963257

####################
#     IMPORTS
####################
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

import sys

# --- constants ---
Y_AXIS = 1
X_AXIS = 0
FIT_BAD_INPUT = "Bad input to 'fit' method"


def global_score(X: np.array, y: np.array, y_hat: np.array) -> dict:
    """
    Global function to use for all the classes
    Given an unlabeled test set X, and the true labels y,
    of this test set, returns a dictionary with the following fields:
    number of samples in the test set, error rate, accuracy, false positive rate,
    true positive rate, precision, recall (True positive rate)
    :param X: Data to predict
    :param y: The real values of the labels {-1, 1}
    :param y_hat: the predicted vec {-1, 1}
    :return: dict
    """
    num_samples = X.shape[Y_AXIS]
    err_rate = np.sum(y_hat != y) / (num_samples + 0.0)  # (FP + FN) / (P + N)
    accuracy = np.sum(y_hat == y) / (num_samples + 0.0)  # (TP + TN) / (P + N)
    fp = 0
    tp = 0
    for index, item in enumerate(y_hat):
        if item == 1 and y[index] == -1:
            fp += 1
        if item == 1 and y[index] == 1:
            tp += 1
    fpr = fp / np.count_nonzero(((y + np.ones(y.shape[0], dtype=int)) // 2) == 0)
    tpr = tp / np.count_nonzero(y + np.ones(y.shape[0], dtype=int))
    precision = tp / (tp + fp)
    recall = tpr  # https://moodle2.cs.huji.ac.il/nu19/mod/forumng/discuss.php?d=9152
    return {"num_samples": num_samples, "error": err_rate, "accuracy": accuracy,
            "FPR": fpr, "TPR": tpr, "precision": precision, "recall": recall
            }


class Perceptron:
    """
    This is the the class implements the Perceptron algorithm.
    """
    model = []

    def fit(self, X: np.array, y: np.array):
        """
        This method learns the parameters of the model and
        stores the trained model (namely, the variables that define
        the hypothesis chosen) in self.model.
        :param X: The data vector R^dxm. (where d is the number of features and m is number of samples)
        :param y: The lables vector.
        :return: nothing.
        """
        # check input validity
        if X.shape[Y_AXIS] != y.shape[X_AXIS]:
            print(FIT_BAD_INPUT, file=sys.stderr)
            return
        X = np.vstack((np.ones(X.shape[Y_AXIS]), X)).transpose()  # stacking those ones
        w = np.zeros(X.shape[Y_AXIS])
        # Batch Perceptron Algorithm
        while True:
            updated_w_flag = False
            for index in range(len(y)):
                if (y[index] * (w @ X[index])) <= 0:
                    w = w + (y[index] * X[index])
                    updated_w_flag = True
            if not updated_w_flag:
                self.model = w
                return

    def predict(self, X: np.array) -> np.array:
        """
        Given an unlabeled test set X, predicts the label of each sample.
        :param X: Unlabeled test set.
        :return: A vector of predicted lables y.
        """
        non_homogeneous = np.vstack((np.ones(X.shape[Y_AXIS]), X))
        v_func = np.vectorize(lambda x: 1 if x >= 0 else -1)  # vector function to map values to labels
        return v_func(non_homogeneous.transpose() @ self.model)

    def score(self, X: np.array, y: np.array) -> dict:
        """
        Given an unlabeled test set X, and the true labels y,
        of this test set, returns a dictionary with the following fields:
        number of samples in the test set, error rate, accuracy, false positive rate,
        true positive rate, precision, recall (True positive rate)
        :param X:
        :param y:
        :return: dict
        """
        y_hat = self.predict(X)
        return global_score(X, y, y_hat)


class LDA:
    """
    This is the the class implements the LDA classifier.
    """
    model = {}

    def fit(self, X: np.array, y: np.array):
        """
        This method learns the parameters of the model and
        stores the trained model (namely, the variables that define
        the hypothesis chosen) in self.model.
        :param X: The data vector R^dxm. (where d is the number of features and m is number of samples)
        :param y: The lables vector.
        :return: nothing.
        """
        m = X.shape[Y_AXIS]
        features_avgs = (np.add.accumulate(X, axis=Y_AXIS) / m)[:, -1]
        x_t_centered = X.transpose() - np.tile(features_avgs, (m, 1))
        cov_mat = (x_t_centered.transpose() @ x_t_centered) / (m - 1)
        mu_1 = (1 / m) * (np.add.accumulate(X[:, y == 1], axis=Y_AXIS)[:, -1])
        mu_m1 = (1 / m) * (np.add.accumulate(X[:, y == -1], axis=Y_AXIS)[:, -1])
        prob_y1 = np.count_nonzero(y == 1) / m
        prob_ym1 = np.count_nonzero(y == -1) / m
        self.model = {"cov_mat": cov_mat, "mu_1": mu_1, "mu_m1": mu_m1,
                      "prob_y1": prob_y1, "prob_ym1": prob_ym1}

    def predict(self, X: np.array) -> np.array:
        """
        Given an unlabeled test set X, predicts the label of each sample.
        :param X: Unlabeled test set.
        :return: A vector of predicted lables y.
        """
        X = X.transpose()
        res = []
        cov_mat = self.model['cov_mat']
        cov_mat_inv = np.linalg.inv(cov_mat)
        mu1 = self.model['mu_1']
        mum1 = self.model['mu_m1']
        middle_y1 = (0.5 * mu1 @ cov_mat_inv @ mu1)
        middle_ym1 = (0.5 * mum1 @ cov_mat_inv @ mum1)
        # y
        prob_y1 = self.model['prob_y1']
        prob_ym1 = self.model['prob_ym1']
        for x in X:
            if ((x @ cov_mat_inv @ mu1) - middle_y1 + np.log(prob_y1)) >=\
                    ((x @ cov_mat_inv @ mum1) - middle_ym1 + np.log(prob_ym1)):
                res.append(1)
            else:
                res.append(-1)
        return res

    def score(self, X: np.array, y: np.array) -> dict:
        """
        Given an unlabeled test set X, and the true labels y,
        of this test set, returns a dictionary with the following fields:
        number of samples in the test set, error rate, accuracy, false positive rate,
        true positive rate, precision, recall (True positive rate)
        :param X:
        :param y:
        :return: dict
        """
        y_hat = self.predict(X)
        return global_score(X, y, y_hat)


class SVM:
    """
    SVM classifier
    """
    model = None

    def fit(self, X: np.array, y: np.array):
        """
        This method learns the parameters of the model and
        stores the trained model (namely, the variables that define
        the hypothesis chosen) in self.model.
        :param X: The data vector R^dxm. (where d is the number of features and m is number of samples)
        :param y: The lables vector.
        :return: nothing.
        """
        svm = SVC(C=1e10, kernel='linear')
        svm.fit(X.transpose(), y)
        self.model = svm

    def predict(self, X: np.array) -> np.array:
        """
        Given an unlabeled test set X, predicts the label of each sample.
        :param X: Unlabeled test set.
        :return: A vector of predicted lables y.
        """
        return self.model.predict(X.transpose())

    def score(self, X: np.array, y: np.array) -> dict:
        """
        Given an unlabeled test set X, and the true labels y,
        of this test set, returns a dictionary with the following fields:
        number of samples in the test set, error rate, accuracy, false positive rate,
        true positive rate, precision, recall (True positive rate)
        :param X:
        :param y:
        :return: dict
        """
        y_hat = self.predict(X)
        df = global_score(X, y, y_hat)
        df['accuracy'] = self.model.score(X.transpose(), y)
        return df


class Logistic:
    """
    Logistic classifier
    """
    model = None

    def fit(self, X: np.array, y: np.array):
        """
        This method learns the parameters of the model and
        stores the trained model (namely, the variables that define
        the hypothesis chosen) in self.model.
        :param X: The data vector R^dxm. (where d is the number of features and m is number of samples)
        :param y: The lables vector.
        :return: nothing.
        """
        self.model = LogisticRegression(solver='liblinear')
        self.model.fit(X, y)

    def predict(self, X: np.array) -> np.array:
        """
        Given an unlabeled test set X, predicts the label of each sample.
        :param X: Unlabeled test set.
        :return: A vector of predicted lables y.
        """
        return self.model.predict(X)

    def score(self, X: np.array, y: np.array) -> dict:
        """
        Given an unlabeled test set X, and the true labels y,
        of this test set, returns a dictionary with the following fields:
        number of samples in the test set, error rate, accuracy, false positive rate,
        true positive rate, precision, recall (True positive rate)
        :param X:
        :param y:
        :return: dict
        """
        y_hat = self.predict(X)
        return global_score(X, y, y_hat)


class DecisionTree:
    model = None

    def fit(self, X: np.array, y: np.array):
        """
        This method learns the parameters of the model and
        stores the trained model (namely, the variables that define
        the hypothesis chosen) in self.model.
        :param X: The data vector R^dxm. (where d is the number of features and m is number of samples)
        :param y: The lables vector.
        :return: nothing.
        """
        X = X.transpose()
        self.model = DecisionTreeClassifier(max_depth=1)
        self.model.fit(X, y)

    def predict(self, X: np.array) -> np.array:
        """
        Given an unlabeled test set X, predicts the label of each sample.
        :param X: Unlabeled test set.
        :return: A vector of predicted lables y.
        """
        return self.model.predict(X)

    def score(self, X: np.array, y: np.array) -> dict:
        """
        Given an unlabeled test set X, and the true labels y,
        of this test set, returns a dictionary with the following fields:
        number of samples in the test set, error rate, accuracy, false positive rate,
        true positive rate, precision, recall (True positive rate)
        :param X:
        :param y:
        :return: dict
        """
        y_hat = self.predict(X)
        return global_score(X, y, y_hat)
