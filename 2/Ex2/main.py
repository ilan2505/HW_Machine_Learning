import AdaBoost
import numpy as np
import Utility
import Line
from itertools import combinations, permutations
import Circle
import matplotlib.pyplot as plt


def __run_adaboost(data: np.ndarray, iteration: int, method: str = "line") -> None:
    rules = []
    if method == "line":
        for p1, p2 in combinations(data, 2):
            # add all possible lines to the list of rules with red color side and blue color side
            rules.append(Line.Line(p1[0:2], p2[0:2], 1))  # red color side
            rules.append(Line.Line(p1[0:2], p2[0:2], -1))  # blue color side
    if method == "circle":
        for p1, p2 in permutations(data, 2):
            # add all possible circles to the list of rules with red color side and blue color side
            rules.append(Circle.Circle(p1[0:2], p2[0:2], 1))  # red color side
            rules.append(Circle.Circle(p1[0:2], p2[0:2], -1))  # blue color side
    print("AdaBoost with", method+"s:")
    np.random.seed(42)
    avg_empirical_error = np.zeros(8)
    avg_true_error = np.zeros(8)
    for i in range(iteration):
        print("Iteration: ", i+1)
        train, test = Utility.random_train_test_split(data)
        ada_boost = AdaBoost.AdaBoost(rules)
        ada_boost.fit(train[:, 0:2], train[:, 2], 8)  # run 8 iterations
        for j in range(8):
            empirical_error = np.sum(ada_boost.predict(train[:, 0:2], j + 1) != train[:, 2])
            true_error = np.sum(ada_boost.predict(test[:, 0:2], j + 1) != test[:, 2])
            avg_empirical_error[j] += empirical_error / len(train)
            avg_true_error[j] += true_error / len(test)
    avg_empirical_error /= iteration
    avg_true_error /= iteration
    print("Results for method:", method)
    print("Average empirical error: ", avg_empirical_error)
    print("Average true error: ", avg_true_error)
    # plot the average empirical error and average true error in same graph of Circle or Line
    plt.title("Average empirical error and average true error of " + method)
    # add title to axis
    plt.xlabel("Numbers of " + method + "s")
    plt.ylabel("Error")
    # color the graph of empirical error with red and true error with blue
    plt.plot(range(1, 9), avg_empirical_error, 'r', label="Empirical error")
    plt.plot(range(1, 9), avg_true_error, 'b', label="True error")
    # add legend to the plot
    plt.legend()
    plt.show()


if __name__ == '__main__':
    data = Utility.read_data(Utility.FILE_NAME)
    #data[:, -1] = np.where(data[:, -1] == 0, -1, data[:, -1])
    __run_adaboost(data, 50, "line")
    #__run_adaboost(data, 50, "circle")

