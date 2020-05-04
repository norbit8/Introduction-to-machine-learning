# Programmed by Yoav Levy
# ID 314963257

####################
#     IMPORTS
####################
import numpy as np
import pandas as pd
from pandas import DataFrame
import sys

# --- constants ---
Y_AXIS = 1
X_AXIS = 0
FIT_BAD_INPUT = "Bad input to 'fit' method"


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
        print(non_homogeneous)
        v_func = np.vectorize(lambda x: 1 if x >= 0 else -1)  # vector function to map values to labels
        return v_func(non_homogeneous.transpose() @ self.model)

    def score(self, X: np.array, y: np.array) -> dict:
        """
        Given an unlabeled test set X, and the true lables y,
        of this test set, returns a dictionary with the following fields:
        number of samples in the test set, error rate, accuracy, false positive rate,
        true positive rate, precision, recall (True positive rate)
        :param X:
        :param y:
        :return:
        """
        num_samples = X.shape[Y_AXIS]
        y_hat = self.predict(X)
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


class LDA:
    """
    This is the the class implements the LDA classifier.
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
        m = X.shape[Y_AXIS]
        d = X.shape[X_AXIS]
        features_avgs = (np.add.accumulate(X, axis=1) / m)[:, -1]
        x_t_centered = X.transpose() - np.tile(features_avgs, (m, 1))
        cov_mat = (x_t_centered.transpose() @ x_t_centered) / (m - 1)
        mu_1 = (1 / m) * (np.add.accumulate(X[y == 1], axis=1)[:, -1])
        mu_m1 = (1 / m) * (np.add.accumulate(X[y == -1], axis=1)[:, -1])
        prob_y1 = np.count_nonzero(y == 1) / m
        prob_ym1 = np.count_nonzero(y == -1) / m

    def predict(self, X: np.array) -> np.array:
        pass

    def score(self, X: np.array, y: np.array) -> dict:
        pass


class SVM:
    model = []

    def fit(self, X: np.array, y: np.array):
        pass

    def predict(self, X: np.array) -> np.array:
        pass

    def score(self, X: np.array, y: np.array) -> dict:
        pass


class Logistic:
    model = []

    def fit(self, X: np.array, y: np.array):
        pass

    def predict(self, X: np.array) -> np.array:
        pass

    def score(self, X: np.array, y: np.array) -> dict:
        pass


class DecisionTree:
    model = []

    def fit(self, X: np.array, y: np.array):
        pass

    def predict(self, X: np.array) -> np.array:
        pass

    def score(self, X: np.array, y: np.array) -> dict:
        pass


if __name__ == '__main__':
# data = pd.read_csv("/home/mercydude/University/"
#                    "semester05/Introduction to machine learning/projects/project_03/tests/iris.csv")
# data.loc[data.flower == "Iris-setosa", 'flower'] = 0
# data.loc[data.flower == "Iris-virginica", 'flower'] = 1
# data = data.sample(frac=1)
# X = data.iloc[:, :4].to_numpy().transpose()
# y = data.iloc[:, 4:].to_numpy()
# print(X.shape)
# print(y.shape)


## PERCEPTRON BAISC TESTS
# x = np.array([[1, 3], [1, 4], [2, 4], [4, 1], [5, 1]]).transpose()
# y = np.array([1, 1, 1, -1, -1])
#
# perce = Perceptron()
# perce.fit(x, y)
# print(perce.model)
# testing_data = np.array([[1, 2]]).transpose()
# print(testing_data.shape)
# print("PASSED") if (perce.predict(testing_data) == 1)[0] else print("FAILED")
