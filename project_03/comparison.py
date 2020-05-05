import numpy as np
from models import *
from plotnine import *
import pandas as pd
from pandas import DataFrame


def draw_points(m):
    np.random.seed(0)
    mu = np.array([0, 0])  # mean vector of zeros
    scale = np.eye(2, 2)
    samples = np.random.multivariate_normal(mu, scale, m).transpose()
    w = np.array([0.3, -0.5])
    b = 0.1
    y = np.sign((w @ samples) + b)
    return (samples, y)


if "__main__" == __name__:
    nums = [5, 10, 15, 25, 70]
    beta1 = 0.3 / 0.5
    beta0 = 0.1 / 0.5
    line = np.linspace(-3, 3, 10)
    hp = ((beta1 * line) + beta0)  # TRUE HYPOTHESIS
    perceptron = Perceptron()
    svm = SVM()
    for m in nums:
        break
        x, y = draw_points(m)
        x1_blue = x[0][y == 1]
        x2_blue = x[1][y == 1]
        x1_orange = x[0][y == -1]
        x2_orange = x[1][y == -1]
        # perceptron
        perceptron.fit(x, y)
        per_x = perceptron.model[1]
        per_y = perceptron.model[2]
        per_b = perceptron.model[0]
        hp2 = ((-per_x / per_y * line) - (per_b / per_y))
        # SVM
        svm.fit(x, y)
        svm_xy = svm.model.coef_[0]
        svm_b = svm.model.intercept_
        svm_x = svm_xy[0]
        svm_y = svm_xy[1]
        hp3 = ((-svm_x / svm_y * line) - (svm_b / svm_y))
        # DATA FRAME
        df1 = DataFrame({'x1_blue': x1_blue, 'x2_blue': x2_blue, 'tags': np.ones(len(x2_blue))})  # Blue dots
        df2 = DataFrame(
            {'x1_orange': x1_orange, 'x2_orange': x2_orange, 'tags': -1 * np.ones(len(x2_orange))})  # Orange dots
        df3 = DataFrame(
            {'x': line, "hp": hp, "name": ['True hypothesis hyperplane'] * len(hp)})  # True hypothesis HyperPlane
        df4 = DataFrame({'x': line, "hp2": hp2, "name": ['Perceptron hyperplane'] * len(hp)})  # Perceptron HyperPlane
        df5 = DataFrame({'x': line, "hp3": hp3, "name": ['SVM HyperPlane hyperplane'] * len(hp)})  # SVM HyperPlane

        # PLOT
        p = (ggplot() + \
             geom_point(aes(x='x1_blue', y='x2_blue', fill='factor(tags)'), stroke=0, data=df1, color="blue", size=2) + \
             geom_point(aes(x='x1_orange', y='x2_orange', fill='factor(tags)'), stroke=0, data=df2, color="orange",
                        size=2) + \
             scale_fill_manual(values=["blue", "orange"]) + \
             geom_smooth(aes(x='x', y='hp', colour="name"), data=df3, method="lm") + \
             geom_smooth(aes(x='x', y='hp2', colour="name"), data=df4, method="lm") + \
             geom_smooth(aes(x='x', y='hp3', colour="name"), data=df5, method="lm") + \
             scale_colour_manual(name="Hyperplanes", values=("red", "black", "purple")) + \
             labs(x="Feature 1 ($x_1$)", y="Feature 2 ($x_2$)", fill="Labels",
                  title=f"Question 9: Testing SVM algorithm VS"
                        f" Perceptron (number of samples:{m})"))
        print(p)
        ggsave(p, f"plot{m}.png", width=4, height=4)
        # TODO: clean the code ^^ create functions FFS
        # -------------------------------
    lda = LDA()
    k = 10000
    for m in nums:
        print(f"--- TESTING ON {m} trained samples ---")
        x_train, y_train = draw_points(m)
        perceptron.fit(x_train, y_train)
        svm.fit(x_train, y_train)
        lda.fit(x_train, y_train)

        perceptron_accuracy_sum, svm_accuracy_sum, lda_accuracy_sum = (0, 0, 0)
        for index in range(500):
            print(index)
            x_test, y_test = draw_points(k)
            score_perceptron = perceptron.score(x_test, y_test)
            score_svm = svm.score(x_test, y_test)
            score_lda = lda.score(x_test, y_test)
            perceptron_accuracy_sum += score_perceptron['accuracy']
            svm_accuracy_sum += score_svm['accuracy']
            lda_accuracy_sum += score_lda['accuracy']

        print("Perceptron avg accuracy: ", perceptron_accuracy_sum / 500)
        print("SVM avg accuracy: ", svm_accuracy_sum / 500)
        print("LDA avg accuracy: ", lda_accuracy_sum / 500)