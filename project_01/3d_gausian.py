# --------------------
# cs user: yoav
# full name: Yoav Levy
# id: 314963257
# --------------------
# ----- imports: -----
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import qr
# ---------------------


def get_orthogonal_matrix(dim):
    H = np.random.randn(dim, dim)
    Q, R = qr(H)
    return Q


def plot_3d(x_y_z: np.array, title: str):
    '''
    plot points in 3D
    :param x_y_z: the points. numpy array with shape: 3 X num_samples (first dimension for x, y, z
    coordinate)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_y_z[0], x_y_z[1], x_y_z[2], s=1, marker='.', depthshade=False)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title)
    fig.show()


def plot_2d(x_y, title):
    '''
    plot points in 2D
    :param x_y_z: the points. numpy array with shape: 2 X num_samples (first dimension for x, y
    coordinate)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_y[0], x_y[1], s=1, marker='.')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    fig.show()


def estimator_function(data):
    """
    This function is the mean estimator function.
    :param data: Coin tosses data.
    :return: The mean of tosses numbers.
    """
    values = []
    for index in range(len(data)):
        values.append(np.mean(data[:index + 1]))
    return values


def plot_q16(data):
    """
    This function plots the sample mean of a coin toss as a function of the number of tosses (m)
    as requested in question number 16 (a).
    :param data: 5d vector of vectors where the data of each batch is.
    :return: nothing.
    """
    data1 = estimator_function(data[0])
    data2 = estimator_function(data[1])
    data3 = estimator_function(data[2])
    data4 = estimator_function(data[3])
    data5 = estimator_function(data[4])
    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.plot(range(1, 1001), data1, label="sequence 1")  # plot 1
    ax.plot(range(1, 1001), data2, label="sequence 2")  # plot 2
    ax.plot(range(1, 1001), data3, label="sequence 3")  # plot 3
    ax.plot(range(1, 1001), data4, label="sequence 4")  # plot 4
    ax.plot(range(1, 1001), data5, label="sequence 5")  # plot 5
    ax.set_xlabel("Toss numbers (m)")
    ax.set_ylabel("Sample mean (estimator)")
    ax.legend()
    ax.set_ylim(ymin=0, ymax=1)
    ax.set_xlim(xmin=0, xmax=1000)
    plt.grid(color='#999999', linestyle='--', alpha=0.3)
    ax.set_title("Q16: 5 independent sequences of 1,000 tosses and their mean")
    fig.show()

def print_cov_mat(data):
    cov_mat = np.matmul(np.array([[1/(population - 1), 0, 0], [0, 1/(population - 1), 0],[0, 0, 1/(population - 1)]]),
                        np.matmul(data, np.transpose(data)))
    print(cov_mat)

def question_11(data):
    """
    Question 11, just plotting the given data.
    :param data: data
    :return: nothing
    """
    plot_3d(data, "Q11: Random points gen by the identity matrix as the cov matrix")  # Q11


def question_12(data):
    """
    Question 12 linear transform the data, plot it, and show the newly created cov matrix.
    :param data: data
    :return: data after linear transformation
    """
    s_mat = np.array([[0.1, 0, 0], [0, 0.5, 0], [0, 0, 2]])
    new_data = np.matmul(s_mat, data)
    plot_3d(new_data, "Q12: Linear Transformed the prev data")
    print("------ Covariance Matrix (QUESTION 12) ------")
    print_cov_mat(new_data)
    return new_data


def question_13(data):
    """
    Question 13, create an orthogonal matrix, applay on the given data (already linear transformed)
    plot it, and show the cov matrix. (also prints the transformed data)
    :param data: Linear transformed data.
    :return: nothing.
    """
    print("------ Randomly created orthogonal Matrix ------")
    orthogonal_mat = get_orthogonal_matrix(3) # Randomly orthogonal matrix
    print(orthogonal_mat)
    new_data = np.matmul(orthogonal_mat, data)
    plot_3d(new_data, "Q13: Applied orthogonal matrix")
    print("------ Covariance Matrix (AFTER ORTHGONAL TRANSFORMATION) ------")
    print_cov_mat(new_data)


def question_14(data):
    """
    Question 14, projection of the data to the x,y plane.
    and prints the result, and plotting the data.
    :param data: data.
    :return: nothing.
    """
    print("------ Projection of the data to the x, y axes (DATA) ------")
    proj_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    two_d_data = np.matmul(proj_mat, data)
    print(two_d_data[:-1])  # "Elisha Kipnis" said we should plot in 2d if we project to 2d plane
    # (https://moodle2.cs.huji.ac.il/nu19/mod/forumng/discuss.php?d=7931)
    plot_2d(two_d_data[:-1], "Q14: Projection of the data to the x, y axes.")


def question_15(data):
    """
    Question 15, projection of the data which 0.1 > z > -0.4 to the x,y plane.
    and prints the result, and plotting the data.
    :param data:
    :return: nothing.
    """
    print("------ Question 15 0.1 > z > -0.4 (DATA) ------")
    x = []
    y = []
    bool_ind = (-0.4 < data[2]) & (data[2] < 0.1)
    for index, item in enumerate(data[0]):
        if bool_ind[index]:
            x.append(item)
            y.append(data[1][index])
    data_01zm04 = np.array([x, y])
    print(data_01zm04)
    plot_2d(data_01zm04, "Q15: Projection to x, y after filtering dots 0.1 > z > -0.4")


def question_16_a(data):
    """
    Question 16 a, plotting the experiment.
    :param data: data of 5 batches of 1000 coin tosses.
    :return: nothing.
    """
    plot_q16(data[:5])


def true_false_inequality(data, epsilons):
    """

    :param data: Data.
    :return: The variance value.
    """
    values = []
    for eps in epsilons:
        for index in range(len(data)):
            values.append(abs(np.mean(data[:index + 1]) - 0.25) >= eps)
    return values


def question_16_b(data, epsilons):
    """
    Question 16 b,
    :param data:
    :return:
    """
    variance = 0.1875
    for eps in epsilons:
        fig, ax = plt.subplots()  # Create a figure containing a single axes.
        chebyshev = [min(variance/(x*(eps**2)), 1) for x in range(1, 1001)]
        hoeffding = [min(2*math.exp(-2*x*(eps**2)), 1) for x in range(1, 1001)]
        ax.plot(range(1, 1001), chebyshev, label="chebyshev")  # plot 1
        ax.plot(range(1, 1001), hoeffding, label="hoeffding")  # plot 1
        print("eps="+str(eps), "hoeffding=", hoeffding)
        print("chebyshev=", chebyshev)
        ax.set_xlabel("Toss numbers (m)")
        ax.set_ylabel("Upper Bound")
        ax.legend()
        ax.set_ylim(ymin=0, ymax=1.01)
        ax.set_xlim(xmin=0, xmax=1000)
        plt.grid(color='#999999', linestyle='--', alpha=0.3)
        ax.set_title("Q16b: epsilon = " + str(eps))
        fig.show()

def question_16_c(data):
    """

    :param data:
    :return:
    """
    pass


if __name__ == "__main__":
    # ----------- settings -------------
    population = 50000
    mean = [0, 0, 0]
    cov = np.eye(3)
    data = np.random.multivariate_normal(mean, cov, population).T
    np.set_printoptions(suppress=True)
    # ----------------------------------
    # question_11(data)
    # transformed_data = question_12(data)
    # question_13(transformed_data)
    # question_14(data)
    # question_15(data)
    # ----- Question no.16 -----
    data = np.random.binomial(1, 0.25, (5, 1000))
    epsilon = [0.5, 0.25, 0.1, 0.01, 0.001]
    # question_16_a(data)
    # question_16_b(data[0], epsilon)
    tf = []
    for seq in range(len(data)):
        tf.append(true_false_inequality(data[seq], epsilon))
    print(np.array(tf))