"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

Author: Gad Zalcberg
Date: February, 2019

"""
from plotnine import *
from ex4_tools import *
from pandas import DataFrame
from tqdm import tqdm


class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = WL
        self.T = T
        self.h = [None]*T     # list of base learners
        self.w = np.zeros(T)  # weights
        self.D = []

    def train(self, X, y):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        Train this classifier over the sample (X,y)
        After finish the training return the weights of the samples in the last iteration.
        """
        num_samples = X.shape[0]
        self.D = np.array([1 / num_samples] * (num_samples))
        for i in tqdm(range(self.T), "Training progress"):
            self.h[i] = self.WL(self.D, X, y)
            h_t = self.h[i].predict(X)
            err_t = np.dot(self.D, y != h_t)
            self.w[i] = 0.5 * np.log((1 - err_t) / float(err_t))
            self.D = self.D * np.exp(-self.w[i]*y*h_t) / (self.D @ np.exp(-self.w[i]*y*h_t))

    def predict_generator(self, max_t, X):
        i = 0
        while i != max_t:
            yield self.h[i].predict(X)
            i += 1

    def predict(self, X, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: y_hat : a prediction vector for X. shape=(num_samples)
        Predict only with max_t weak learners,
        """
        sum = 0
        for index, vec in enumerate(self.predict_generator(max_t, X)):
            sum += self.w[index] * vec
        return np.sign(sum)
        # s = [f.predict(X) for f in self.h[:max_t]]
        # return np.sign(self.w[:max_t] @ s)

    def error(self, X, y, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: error : the ratio of the correct predictions when predict only with max_t weak learners (float)
        """
        y_hat = self.predict(X, max_t)
        return np.count_nonzero(y_hat != y) / y.shape[0]

    def training_error(self, X, y, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: error : the ratio of the correct predictions when predict only with max_t weak learners (float)
        """
        y_hat = self.predict(X, max_t)
        return np.count_nonzero(y_hat != y) / y.shape[0]


def plot_errors(noise):
    """
    In question 10 we plot graph
    :return: nothing
    """
    g = (ggplot(DataFrame({'T': range(0, 501), 'testing error': testing_error, 'training error': training_error,
                           'test': 'test', 'train': 'train'})) +
         geom_line(aes(x='T', y='testing error', color='test')) +
         geom_line(aes(x='T', y='training error', color='train')) +
         scale_color_manual(name="Error type", values=("red", "blue")) +
         labs(title=f"Prediction error vs number of weak learners (noise = {noise})", x="T - number of weak learners",
             y="Prediction error"))
    print(g)


def plot_decision_graphs():
    """
    In question 11 we plot the decision boundaries graph
    :return: nothing
    """
    for i, t in enumerate(tqdm([5, 10, 50, 100, 200, 500], "Graphs progress")):
        plt.subplot(2, 3, i+1)
        decision_boundaries(ab, X_test, y_test, t)
    plt.show()


def plot_min_t():
    """
    Plots the training data with the classifier that minimizes the testing data.
    :return: nothing
    """
    T_hat = np.argmin(testing_error)
    decision_boundaries(ab, X, y, T_hat, 8)
    plt.title(f'T = {T_hat}, The error is: {testing_error[T_hat]}')
    plt.show()


def init_data(noise):
    """
    Inits the data, which means it generate data and train over it
    also it sets the training and testing error lists.
    :param noise: The noise of the generated data.
    :return: _X, _y, _ab, _X_test, _y_test, _testing_error, _training_error
    """
    _X, _y = generate_data(5000, noise)  # training samples
    _ab = AdaBoost(DecisionStump, 500)
    _ab.train(_X, _y)  # training the model
    _X_test, _y_test = generate_data(200, noise)  # testing samples
    # --------------------------------
    _training_error, _testing_error = [], []
    for max_t in range(0, 501):
        _testing_error.append(_ab.error(_X_test, _y_test, max_t))
        _training_error.append(_ab.training_error(_X, _y, max_t))
    return _X, _y, _ab, _X_test, _y_test, _testing_error, _training_error


def plot_training_weighted_dots():
    """
    Question 13
    plots the training set where the dots are weighted.
    """
    decision_boundaries(ab, X, y, 499, (ab.D / np.max(ab.D)) * 10)
    plt.title(f'Plot of the training set with the weights\n'
              f' of the last iteration reflected by the size of each dot')
    plt.show()


if '__main__' == __name__:
    noise = 0
    X, y, ab, X_test, y_test, testing_error, training_error = init_data(noise)
    plot_errors(noise)  # Q10
    plot_decision_graphs()  # Q11
    plot_min_t()  # Q 12
    # Q13 -
    plot_training_weighted_dots()
    # ----------------------
    noise = 0.01
    X, y, ab, X_test, y_test, testing_error, training_error = init_data(noise)
    plot_errors(noise)  # Q10
    plot_decision_graphs()  # Q11
    plot_min_t()  # Q 12
    # Q13 -
    plot_training_weighted_dots()
    # ----------------------
    noise = 0.4
    X, y, ab, X_test, y_test, testing_error, training_error = init_data(noise)
    plot_errors(noise)  # Q10
    plot_decision_graphs()  # Q11
    plot_min_t()  # Q 12
    # Q13 -
    plot_training_weighted_dots()
