import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import qr
from scipy.stats import special_ortho_group

population = 50000
mean = [0, 0, 0]
cov = np.eye(3)
x_y_z = np.random.multivariate_normal(mean, cov, population).T


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

    :param data:
    :return:
    """
    values = []
    for index in range(len(data)):
        values.append(np.mean(data[:index + 1]))
    return values

def plot_q16(data):
    """

    :param data:
    :return:
    """
    data1 = estimator_function(data[0])
    data2 = estimator_function(data[1])
    data3 = estimator_function(data[2])
    data4 = estimator_function(data[3])
    data5 = estimator_function(data[4])
    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.plot(range(1, 1001), data1, label="sequence 1")  # Plot some data on the axes.
    ax.plot(range(1, 1001), data2, label="sequence 2")  # Plot some data on the axes.
    ax.plot(range(1, 1001), data3, label="sequence 3")
    ax.plot(range(1, 1001), data4, label="sequence 4")
    ax.plot(range(1, 1001), data5, label="sequence 5")
    ax.set_xlabel("Toss number")
    ax.set_ylabel("Sample mean (estimator)")
    ax.legend()
    ax.set_ylim(ymin=0, ymax=1)
    ax.set_xlim(xmin=0, xmax=1000)
    plt.grid(color='#999999', linestyle='--', alpha=0.3)
    ax.set_title("Q16: 5 independent sequences of 1,000 tosses and their mean")
    fig.show()


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    # data = x_y_z
    # plot_3d(x_y_z, "Q11: Random points gen by the identity matrix as the cov matrix")  # Q11
    # s_mat = np.array([[0.1, 0, 0], [0, 0.5, 0], [0, 0, 2]])
    # new_data = np.matmul(s_mat, x_y_z)
    # plot_3d(new_data, "Q12: Linear Transformed the prev data")
    # cov_mat = np.matmul(np.array([[1/(population - 1), 0, 0], [0, 1/(population - 1), 0],[0, 0, 1/(population - 1)]]),
    #                     np.matmul(new_data, np.transpose(new_data)))
    # print("------ Covariance Matrix ------")
    # print(cov_mat)
    # print("------ Orthgonal Matrix ------")
    # orthogonal_mat = special_ortho_group.rvs(3) # Randomly orthogonal matrix
    # print(orthogonal_mat)
    # new_data = np.matmul(orthogonal_mat, new_data)
    # plot_3d(new_data, "Q13: Applied orthogonal matrix")
    # print("------ Covariance Matrix (AFTER ORTHGONAL TRANSFORMATION) ------")
    # cov_mat = np.matmul(np.array([[1/(population - 1), 0, 0], [0, 1/(population - 1), 0],[0, 0, 1/(population - 1)]]),
    #                     np.matmul(new_data, np.transpose(new_data)))
    # print(cov_mat)
    # print("------ Projection of the data to the x, y axes ------")
    # proj_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    # two_d_data = np.matmul(proj_mat, data)
    # print(two_d_data[:-1]) # Elisha Kipnis said we should plot in 2d if we project to 2d plane (https://moodle2.cs.huji.ac.il/nu19/mod/forumng/discuss.php?d=7931)
    # plot_2d(two_d_data[:-1], "Q14: Projection of the data to the x, y axes.")
    # print("------ Question 15 0.1 > z > -0.4 ------")
    # x = []
    # y = []
    # bool_ind = (-0.4 < data[2]) & (data[2] < 0.1)
    # for index, item in enumerate(data[0]):
    #     if bool_ind[index]:
    #         x.append(item)
    #         y.append(data[1][index])
    #
    # data_01zm04 = np.array([x, y])
    # print(data_01zm04)
    # plot_2d(data_01zm04, "Q15: Projection to x, y after filtering dots 0.1 > z > -0.4")
    # ----------------------------------------------------------
    data = np.random.binomial(1, 0.25, (5, 1000))
    epsilon = [0.5, 0.25, 0.1, 0.01, 0.001]
    plot_q16(data[:5])
