# Programmed by Yoav Levy
# ID 314963257

####################
#     IMPORTS
####################
import tensorflow as tf
import numpy as np
from matplotlib import pyplot
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from pandas import DataFrame
from plotnine import *
import time

# --- constants ---
TRAINING_POINTS_NUMBERS = (50, 100, 300, 500)
AVG = 50
IMG_RES_PIXELS = 28


def rearrange_data(X: np.array) -> np.array:
    """
    Given an np array of size m x 28 x 28,
    reshapes it to m x 784 np array
    :param X: array of size m x 28 x 28
    :return: array of size m x 784
    """
    return X.reshape(X.shape[0], IMG_RES_PIXELS * IMG_RES_PIXELS)


def question_12():
    """
    In this question we draw 3 zeros and 3 ones
    :return: nothing
    """
    zeros = (0, 5, 8)
    ones = (1, 2, 3)
    for i in zip(zeros, ones):
        pyplot.imshow(x_train[i[0]])
        pyplot.show()
        pyplot.imshow(x_train[i[1]])
        pyplot.show()


def draw_uniformly(m: int, data: np.array, y_train: np.array) -> tuple:
    """
    Draws m different data points (UNIFORMLY)
    :param m: number of poitns to draw
    :param data: the data to draw from
    :return: the drawn points.
    """
    points = []  # list of uniformly selected points
    for i in range(m):
        while True:
            point = np.random.randint(data.shape[0])
            if point not in points:  # this validates that we wont have the same samples in the drawn points
                break
        points.append(point)
    if 1 not in y_train[points] or 0 not in y_train[points]:
        return draw_uniformly(m, data, y_train)
    else:
        return data[points], y_train[points]


def calculate_accuracy(lr, svm, dct, neigh, data, true_labels) -> tuple:
    """
    This function calculates the accuracies of each learning algorithm
    (average of 50 different accuracies)
    :param lr: Logistic regression algorithm
    :param svm: SVM algorithm
    :param dct: Decision Tree algorithm
    :param neigh: KNN algorithm.
    :param data: The data to run the algos on.
    :param true_labels: True labels with respect to the given data.
    :return: Tuple of all the different accuracies.
    """
    counter = AVG
    sum_accuracy_lr, sum_accuracy_svm, sum_accuracy_dct, sum_accuracy_neigh = 0, 0, 0, 0
    while counter != 0:
        # print("IN WHILE")
        lr_labels = lr.predict(data)
        svm_labels = svm.predict(data)
        dct_labels = dct.predict(data)
        neigh_labels = neigh.predict(data)
        sum_accuracy_lr += np.count_nonzero(lr_labels == true_labels) / true_labels.shape[0]
        sum_accuracy_svm += np.count_nonzero(svm_labels == true_labels) / true_labels.shape[0]
        sum_accuracy_dct += np.count_nonzero(dct_labels == true_labels) / true_labels.shape[0]
        sum_accuracy_neigh += np.count_nonzero(neigh_labels == true_labels) / true_labels.shape[0]
        counter -= 1  # while variable
    return sum_accuracy_lr / AVG, sum_accuracy_svm / AVG, sum_accuracy_dct / AVG, sum_accuracy_neigh / AVG


def plot_accs(df):
    """
    this function plots the accs (QUESTION 14)
    :param df: data frame of all the accuracies
    :return: nothing
    """
    p = (ggplot(df) + \
         geom_point(aes(x='training_size', y='LogisticRegression')) + \
         geom_point(aes(x='training_size', y='SVM')) + \
         geom_point(aes(x='training_size', y='DecisionTree')) + \
         geom_point(aes(x='training_size', y='K_nearest_neighbors')) + \
         geom_smooth(aes(x='training_size', y='LogisticRegression', colour="factor(name1)")) + \
         geom_smooth(aes(x='training_size', y='SVM', colour="factor(name2)")) + \
         geom_smooth(aes(x='training_size', y='DecisionTree', colour="factor(name3)")) + \
         geom_smooth(aes(x='training_size', y='K_nearest_neighbors', colour="factor(name4)")) + \
         scale_color_manual(name="Algorithm", values=("red", "black", "purple", "green")) + \
         labs(x="Trained samples number (m)", y="Accuracy average",
              title=f"Question 14: Accuracies of LR vs SVM vs DC-Tree vs KNN on MNIST") + \
         scale_y_continuous(limits=(0.5, 1)))
    print(p)


def get_df_of_accuracies():
    """
    The main function of question 14, this calculates the
    :return: Data frame of all the accuracies
    """
    for m in TRAINING_POINTS_NUMBERS:
        # print(f"CALCULATING {m}")
        training_data, training_labels = draw_uniformly(m, x_train_r, y_train)
        neigh = KNeighborsClassifier(n_neighbors=m // 3).fit(training_data, training_labels)
        lr.fit(training_data, training_labels)
        svm.fit(training_data, training_labels)
        dct.fit(training_data, training_labels)
        lr_a, svm_a, dct_a, neigh_a = calculate_accuracy(lr, svm, dct, neigh, x_test_r, y_test)
        mean_acc_lr.append(lr_a)
        mean_acc_svm.append(svm_a)
        mean_acc_dct.append(dct_a)
        mean_acc_neigh.append(neigh_a)
    return DataFrame({'training_size': TRAINING_POINTS_NUMBERS, 'LogisticRegression': mean_acc_lr,
                      'SVM': mean_acc_svm, 'DecisionTree': mean_acc_dct, 'K_nearest_neighbors': mean_acc_neigh,
                      'name1': ("Logistic Regression",) * len(TRAINING_POINTS_NUMBERS),
                      'name2': ("SVM",) * len(TRAINING_POINTS_NUMBERS),
                      'name3': ("Decision Tree",) * len(TRAINING_POINTS_NUMBERS),
                      'name4': ("K nearest neighbors",) * len(TRAINING_POINTS_NUMBERS)})


if __name__ == '__main__':
    # init mnist dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    train_images = np.logical_or((y_train == 0), (y_train == 1))
    test_images = np.logical_or((y_test == 0), (y_test == 1))
    x_train, y_train = x_train[train_images], y_train[train_images]
    x_test, y_test = x_test[test_images], y_test[test_images]
    # question 12 printing 3 zeros and 3 ones
    question_12()
    # question 14 pre-processing:
    print("TIMER STARTED")
    timer_before = time.time()
    x_test_r = rearrange_data(x_test)
    x_train_r = rearrange_data(x_train)
    mean_acc_lr, mean_acc_svm, mean_acc_dct, mean_acc_neigh = [], [], [], []
    lr, dct, svm = LogisticRegression(random_state=0), DecisionTreeClassifier(max_depth=1), SVC(C=1,
                                                                                                kernel='linear')
    df = get_df_of_accuracies()
    timer_after = time.time()
    print(timer_after - timer_before)
    plot_accs(df)

