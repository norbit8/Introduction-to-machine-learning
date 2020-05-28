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
        D = np.array([1 / num_samples] * (num_samples))
        for i in range(self.T):
            self.h[i] = self.WL(D, X, y)
            h_t = self.h[i].predict(X)
            err_t = np.dot(D, y != h_t)
            self.w[i] = 0.5 * np.log((1 - err_t) / float(err_t))
            D = D * np.exp(-self.w[i]*y*h_t) / (D @ np.exp(-self.w[i]*y*h_t))

    def predict(self, X, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: y_hat : a prediction vector for X. shape=(num_samples)
        Predict only with max_t weak learners,
        """
        return np.sign(self.w[:max_t] @ [f.predict(X) for f in self.h[:max_t]])

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
        y_hat = ab.predict(X, max_t)
        return np.count_nonzero(y_hat != y) / y.shape[0]

def plot_errors():
    training_error, testing_error = [], []
    for max_t in range(0, 501):
        testing_error.append(ab.error(X_test, y_test, max_t))
        training_error.append(ab.training_error(X, y, max_t))
    g = (ggplot(DataFrame({'T': range(0, 501), 'testing error': testing_error, 'training error': training_error,
                           'test': 'test', 'train': 'train'})) +
         geom_line(aes(x='T', y='testing error', color='test')) +
         geom_line(aes(x='T', y='training error', color='train')) +
         scale_color_manual(name="Error type", values=("red", "blue")) +
         labs(title="Prediction error vs number of weak learners (noise = 0.0)", x="T - number of weak learners",
             y="Prediction error"))
    ggsave(g, "q10.png")


if '__main__' == __name__:
    X, y = generate_data(5000, 0)  # training samples
    ab = AdaBoost(DecisionStump, 500)
    ab.train(X, y)  # training the model
    X_test, y_test = generate_data(200, 0)  # testing samples
    # plot_errors()  # Q10
    for t in [5, 10, 50, 100, 200, 500]:
        decision_boundaries(ab, X_test, y_test, t)
        plt.show()
