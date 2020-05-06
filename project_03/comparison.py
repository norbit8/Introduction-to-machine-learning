# Programmed by Yoav Levy
# ID 314963257

####################
#     IMPORTS
####################
from models import *
from plotnine import *
from pandas import DataFrame

# --- constants ---
TRAINED_NUMBERS = (5, 10, 15, 25, 70)


def draw_points(m: int) -> tuple:
    """
    This function is the implementation of question number 8,
    it draws points from the multivariate normal distribution.
    :param m: How many samples to draw.
    :return: Tuple of samples & labels.
    """
    mu = np.array([0, 0])  # mean vector of zeros
    scale = np.eye(2, 2)
    samples = np.random.multivariate_normal(mu, scale, m).transpose()
    w = np.array([0.3, -0.5])
    b = 0.1
    y = np.sign((w @ samples) + b)
    return (samples, y)


def Draw_question_10_graphs(perceptron_accuracy_avg: list, svm_accuracy_avg: list, lda_accuracy_avg: list):
    """
    This function creates the graph for question number 10
    and prints it.
    :return: nothing.
    """
    perceptron = ("Perceptron",) * 5
    svm = ("SVM",) * 5
    lda = ("LDA",) * 5
    # perceptron_accuracy_avg = (0.9304, 0.9152, 0.9329, 0.9864, 0.9802)
    # svm_accuracy_avg = (0.7771, 0.8407, 0.9805, 0.9585, 0.9834)
    # lda_accuracy_avg = (0.7318, 0.7559, 0.7554, 0.9203, 0.9087)
    df = DataFrame({'trained samples': TRAINED_NUMBERS, 'perceptron': perceptron_accuracy_avg, 'svm': svm_accuracy_avg,
                    'lda': lda_accuracy_avg, 'name1': perceptron, 'name2': svm, 'name3': lda})  # Blue dots
    p = (ggplot(df) + \
         geom_point(aes(x='trained samples', y='perceptron')) + \
         geom_point(aes(x='trained samples', y='svm')) + \
         geom_point(aes(x='trained samples', y='lda')) + \
         geom_smooth(aes(x='trained samples', y='perceptron', colour="factor(name1)")) + \
         geom_smooth(aes(x='trained samples', y='lda', colour="factor(name2)")) + \
         geom_smooth(aes(x='trained samples', y='svm', colour="factor(name3)")) + \
         scale_color_manual(name="Algorithm", values=("red", "black", "purple")) + \
         labs(x="Trained samples number (m)", y="Accuracy average",
              title=f"Question 10: Testing Perceptron VS SVM algorithm VS"
                    f"  LDA") + \
         scale_y_continuous(limits=(0.5, 1)))
    print(p)


def Plot_9(df1: DataFrame, df2: DataFrame, df3: DataFrame, df4: DataFrame, df5: DataFrame, m: int):
    """
    This function plots the above dataframes.
    :param df1: Is the blue dots
    :param df2: Is the orange dots
    :param df3: True hypothesis HyperPlane
    :param df4: Perceptron HyperPlane
    :param df5: SVM HyperPlane
    :param m: The number of samples.
    :return: nothing
    """
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


def Question_9_hyperplanes_graphs():
    """
    Implementation of question 9
    :return: nothing.
    """
    beta1, beta0 = 0.3 / 0.5,  0.1 / 0.5
    line = np.linspace(-3, 3, 10)
    hp = ((beta1 * line) + beta0)  # TRUE HYPOTHESIS
    for m in TRAINED_NUMBERS:
        x, y = draw_points(m)  # Draws the points
        x1_blue, x2_blue, x1_orange, x2_orange = (x[0][y == 1], x[1][y == 1], x[0][y == -1], x[1][y == -1])
        # perceptron
        perceptron.fit(x, y)
        per_x, per_y, per_b = (perceptron.model[1], perceptron.model[2], perceptron.model[0])
        hp2 = ((-per_x / per_y * line) - (per_b / per_y))
        # SVM
        svm.fit(x, y)
        svm_xy, svm_b = (svm.model.coef_[0], svm.model.intercept_)  # x,y and b
        hp3 = ((-svm_xy[0] / svm_xy[1] * line) - (svm_b / svm_xy[1]))
        # DATA FRAMES
        df1 = DataFrame({'x1_blue': x1_blue, 'x2_blue': x2_blue, 'tags': np.ones(len(x2_blue))})  # Blue dots
        df2 = DataFrame(
            {'x1_orange': x1_orange, 'x2_orange': x2_orange, 'tags': -1 * np.ones(len(x2_orange))})  # Orange dots
        df3 = DataFrame(
            {'x': line, "hp": hp, "name": ['True hypothesis hyperplane'] * len(hp)})  # True hypothesis HyperPlane
        df4 = DataFrame({'x': line, "hp2": hp2, "name": ['Perceptron hyperplane'] * len(hp)})  # Perceptron HyperPlane
        df5 = DataFrame({'x': line, "hp3": hp3, "name": ['SVM HyperPlane hyperplane'] * len(hp)})  # SVM HyperPlane
        # PLOT
        Plot_9(df1, df2, df3, df4, df5, m)


def Get_accuracies()-> tuple:
    """
    Helper function to question 10.
    :return: A tuple contains all the different algorithms accuracies averages.
    """
    k = 10000
    p_a_avg, svm_a_avg, lda_a_avg = [], [], []
    for m in TRAINED_NUMBERS:
        x_train, y_train = draw_points(m)
        perceptron.fit(x_train, y_train)
        svm.fit(x_train, y_train)
        lda.fit(x_train, y_train)
        perceptron_accuracy_sum, svm_accuracy_sum, lda_accuracy_sum = (0, 0, 0)
        for index in range(500):
            x_test, y_test = draw_points(k)
            score_perceptron = perceptron.score(x_test, y_test)
            score_svm = svm.score(x_test, y_test)
            score_lda = lda.score(x_test, y_test)
            perceptron_accuracy_sum += score_perceptron['accuracy']
            svm_accuracy_sum += score_svm['accuracy']
            lda_accuracy_sum += score_lda['accuracy']
        p_a_avg.append(perceptron_accuracy_sum / 500)
        svm_a_avg.append(svm_accuracy_sum / 500)
        lda_a_avg.append(lda_accuracy_sum / 500)

    return p_a_avg, svm_a_avg, lda_a_avg


if "__main__" == __name__:
    perceptron, svm, lda = Perceptron(), SVM(), LDA()
    Question_9_hyperplanes_graphs()  # question 9
    perceptron_accuracy_avg, svm_accuracy_avg, lda_accuracy_avg = Get_accuracies()  # question 10 data
    Draw_question_10_graphs(perceptron_accuracy_avg, svm_accuracy_avg, lda_accuracy_avg)  # Question 10 Graphs
