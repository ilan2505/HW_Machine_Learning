import multiprocessing
from typing import Union

import numpy as np


def calculate_error(rule, data: np.ndarray, y: np.ndarray, weights: np.ndarray):
    predictions = rule.predict(data)
    errors = predictions != y  # Calculate the errors
    error_sum = np.dot(errors, weights)  # Compute the weighted sum of errors
    return error_sum



class AdaBoost:
    def __init__(self, rules: list):
        self.rules = rules
        self.alphas = None
        self.weights = None
        self.best_rule = []
        self.chosen_alphas = []

    def fit(self, data: np.ndarray, y: np.ndarray, iteration: int):  # Train the model with AdaBoost
        # initialize weights
        self.alpha = 0
        self.weights = np.ones(len(data)) / len(data)
        for it in range(iteration):
            err_lst = [np.dot((rule.predict(data) != y), self.weights) for rule in self.rules]
            argmin_index = np.argmin(err_lst)
            if err_lst[argmin_index] == 0:
                self.alpha = 1
                self.best_rule.append(argmin_index)
                self.chosen_alphas.append(self.alpha)
                continue
            else:
                self.alpha = 0.5 * np.log((1 - err_lst[argmin_index]) / err_lst[argmin_index])
            # update weights
            self.weights = self.weights * np.exp(-self.alpha * y * self.rules[argmin_index].predict(data))
            # for i in range(len(self.weights)):
            #     self.weights[i] = self.weights[i]\
            #                       * np.exp(-self.alpha * y[i] * self.rules[argmin_index].predict(data[i]))
            # normalize weights
            self.weights = self.weights / np.sum(self.weights)
            # add the index of the best rule in this iteration to the list of best rules
            self.best_rule.append(argmin_index)
            self.chosen_alphas.append(self.alpha)

    def predict(self, data: np.ndarray, index: int) -> Union[int, np.ndarray]:  # Return the class of the point og Hk
        predictions = np.zeros(len(data))
        for i in range(index):
            predictions += self.chosen_alphas[i] * self.rules[self.best_rule[i]].predict(data)
        return np.where(predictions >= 0, 1, -1)
