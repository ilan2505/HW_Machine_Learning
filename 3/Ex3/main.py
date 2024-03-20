import numpy as np
from KNN import KNN
from Utility import read_data, random_train_test_split, read_data_from_csv

def knn_validation(data: np.ndarray, num_of_iter: int, k: int, distance_lp: float = 2):
    empirical_error = 0
    true_error = 0
    for i in range(num_of_iter):
        knnClassifier = KNN(k=k, distance_lp=distance_lp)
        # split the data into train and test in ratio 1:1
        train_data, test_data = random_train_test_split(data)
        # train the model
        knnClassifier.fit(train_data)
        # predict the labels of the test data
        y_pred_train = knnClassifier.predict(train_data[:, :-1]).astype(int)
        y_pred_test = knnClassifier.predict(test_data[:, :-1]).astype(int)
        # calculate the error of the model on the train and test data
        empirical_error += np.sum(y_pred_train != train_data[:, -1].astype(int)) / len(train_data)
        true_error += np.sum(y_pred_test != test_data[:, -1].astype(int)) / len(test_data)

    # print the average error of the model on the train and test data
    print(f"KNN with k = {k} and p = {distance_lp}: empirical error: {empirical_error / num_of_iter}"
          f", true error: {true_error / num_of_iter}")


if __name__ == '__main__':
    random_seed = 42
    np.random.seed(random_seed)
    print("Haberman data set:")
    data = read_data_from_csv("Haberman.csv.arff")
    for k in {1, 3, 5, 7, 9}:
        for p in {1, 2, np.inf}:
            knn_validation(data, 100, k, p)

    print("===========================================")
    print("circle_separator data set:")
    data = read_data("circle_separator.txt")
    # replace the labels from {-1, 1} to {0, 1}
    data[:, -1] = (data[:, -1] + 1) / 2
    for k in {1, 3, 5, 7, 9}:
        for p in {1, 2, np.inf}:
            knn_validation(data, 100, k, p)


