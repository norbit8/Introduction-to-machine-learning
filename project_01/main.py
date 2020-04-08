import numpy as np
from plotnine import *
from sklearn.linear_model import LinearRegression
from pandas import DataFrame
import matplotlib.pyplot as plt

def main():
    # X = np.arange(1,101).reshape((100,1))
    # y = np.dot(X, [7]) + 3
    # reg = LinearRegression().fit(X, y)
    # coef = np.round(reg.coef_[0])
    # print("The coefficient is "+str(coef))
    # g = ggplot(DataFrame({'X':X.reshape(100),'y':y.reshape(100)})) + \
    #     geom_line(aes(x='X', y='y'), color="blue", size=0.8) + \
    #     ggtitle("f(X) = 7X + 3")
    # print(g)



    labels = ["Arithmetic", "Empty function call", "System call"]
    linux = [0.2, 0.2, 39.8]
    container = [0.2, 0.2, 41]
    vm = [0.9, 47.2, 701.1]
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, linux, width/2, label='linux')
    rects2 = ax.bar(x - width / 3, container, width/2, label='container')
    rects3 = ax.bar(x + width / 3, vm, width/2, label='Virtual Machine')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Nano-Seconds (n) 10^-9')
    ax.set_title('Time measurement of various operations')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yscale('log')
    plt.legend(bbox_to_anchor=(1, 1), loc='best', ncol=1)
    fig.show()


    # np.random.seed(0)  # seed for reproducibility
    # x1 = np.random.randint(10, size=6)  # One-dimensional array
    # x2 = np.random.randint(low=10, size=(3, 4))  # Two-dimensional array
    # x3 = np.random.randint(10, size=(3, 4, 5))  # Three-dimensional array
    # print(np.array([[1, 2, 3], [4, 5, 6]]), "\n --------\n" ,np.array([[1, 2], [4, 5], [6, 7]]))
    # print('-----------------')
    # print(np.matmul(np.array([[1, 2, 3], [4, 5, 6]]), np.array([[1, 2], [4, 5], [6, 7]])))

if __name__ == "__main__":
    main()
