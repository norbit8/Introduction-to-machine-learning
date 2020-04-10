from linear_model import *


def test_01():
    """
    Question 9 test 1.
    Testing the Instagram data from the TA.
    :return:
    """
    x = np.array([[12.3, 57.5, 28.7, 4.2], [7.5, 1.12, 0.8, 14.25]])
    ytag = np.array([46, 4, 11, 75])
    w, singulars = fit_linear_regression(x, ytag)
    ones_vec = np.ones(x.shape[1])  # vectors of ones
    mat_x = np.vstack((ones_vec, x))  # adding ones to the matrix
    print("PASSED TEST 01") if \
    np.allclose(np.dot(mat_x.transpose(), w).tolist(), [44.86995171, 3.83864214, 11.72464346, 75.56676268]) else \
    print("FAILED")


def test_02():
    """
    Question 9 test 2.
    randomly generating matrix and vector testing the linear regression.
    """
    mat = np.random.randint(5, size=(3, 4))
    y = np.random.randint(4, size=4)
    w, singulars = fit_linear_regression(mat, y)
    ones_vec = np.ones(mat.shape[1])  # vectors of ones
    mat_x = np.vstack((ones_vec, mat))  # adding ones to the matrix
    print("PASSED TEST 02") if np.allclose(np.dot(mat_x.transpose().tolist(), w), y) else  print("FAILED TEST 02")


def test_03():
    """
    Question 11 test.
    """
    pred_vec = np.array([4, 5, 6])
    res_vec = np.array([2, 4, 1])
    # print(mse(res_vec, pred_vec))
    print("PASSED TEST 03") if mse(res_vec, pred_vec) == 10 else print("FAILED TEST 03")


if __name__ == "__main__":
    np.random.seed(7)
    print(">>>>> TESTER <<<<<")
    test_01()
    test_02()
    test_03()
    load_data("/home/mercydude/University/semester05/Introduction to machine learning/projects/project_02/kc_house_data.csv")