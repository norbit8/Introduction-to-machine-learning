import numpy as np
from plotnine import *
from sklearn.linear_model import LinearRegression
from pandas import DataFrame

def main():
    # print(np.arange(1,101).reshape((100, 1)))
    X = np.arange(1,101).reshape((100,1))
    y = np.dot(X, [7]) + 3
    print(np.dot(X, [7]) + 3)
    reg = LinearRegression().fit(X, y)
    coef = np.round(reg.coef_[0])
    print("The coefficient is "+str(coef))
    ggplot(DataFrame({'X':X.reshape(100),'y':y.reshape(100)})) + \
        geom_line(aes(x='X', y='y'), color="blue", size=0.8) + \
        ggtitle("f(X) = 7X + 3")

if __name__ == "__main__":
    main()
