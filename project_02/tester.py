from linear_model import *
import sys

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


def test_04(data_location):
    """
    QUESTION 12 tests the loading data function.
    I used diff on the real data and figured out what rows I should eliminate, at the end I ended up with
    just 21606 relevant lines.
    :param data_location: the location of the data
    """
    data = load_data(
        data_location)
    print("PASSED TEST 04") if len(data) == 21606 else print("FAILED TEST 04")

def main():
    try:
        data_location = sys.argv[1]
    except Exception:
        print('USAGE ERROR: please add the "kc_house_data.csv" correct location')
        exit(-1)
    np.random.seed(7)
    print(">>>>> TESTER <<<<<")
    test_01()
    test_02()
    test_03()
    test_04(data_location)
    print(plot_singular_values([3.136, 0.635, 0.534, 0.463, 0.231]))

if __name__ == "__main__":
    main()
