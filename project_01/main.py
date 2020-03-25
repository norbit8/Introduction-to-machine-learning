import numpy as np
from plotnine import *
from sklearn.linear_model import LinearRegression
from pandas import DataFrame

def main():
    X = np.arange(1,101).reshape((100,1))
    y = np.dot(X, [7]) + 3
    reg = LinearRegression().fit(X, y)
    coef = np.round(reg.coef_[0])
    print("The coefficient is "+str(coef))
    g = ggplot(DataFrame({'X':X.reshape(100),'y':y.reshape(100)})) + \
        geom_line(aes(x='X', y='y'), color="blue", size=0.8) + \
        ggtitle("f(X) = 7X + 3")
    print(g)


    # np.random.seed(0)  # seed for reproducibility
    # x1 = np.random.randint(10, size=6)  # One-dimensional array
    # x2 = np.random.randint(low=10, size=(3, 4))  # Two-dimensional array
    # x3 = np.random.randint(10, size=(3, 4, 5))  # Three-dimensional array
    # print(np.array([[1, 2, 3], [4, 5, 6]]), "\n --------\n" ,np.array([[1, 2], [4, 5], [6, 7]]))
    # print('-----------------')
    # print(np.matmul(np.array([[1, 2, 3], [4, 5, 6]]), np.array([[1, 2], [4, 5], [6, 7]])))

if __name__ == "__main__":
    main()
