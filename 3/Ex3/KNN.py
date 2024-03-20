from itertools import combinations
import Utility

import numpy as np

class KNN:
    def __init__(self, k: int, distance_lp: float = 2):
        self.k = k
        self.T = []
        self.distance_lp = distance_lp

    def fit(self, data: np.ndarray):
        # find the minimum distance between red and blue points
        min_distance = np.inf
        for p1, p2 in combinations(data, 2):
            if (not np.array_equal(p1[:-1], p2[:-1])) and p1[-1] != p2[-1]:
                if Utility.lp_distance(p1[:-1], p2[:-1], self.distance_lp) < min_distance:
                    min_distance = Utility.lp_distance(p1[:-1], p2[:-1], self.distance_lp)

        # add random points to the set T from self.X
        self.T.append(data[0])

        for p in data[1:]:
            for t in self.T:
                if Utility.lp_distance(p, t, self.distance_lp) >= min_distance:
                    self.T.append(p)
                    break

    def predict(self, X: np.ndarray) -> np.ndarray:
        y = np.zeros(len(X))
        count = 0
        # find the k nearest neighbors of each point in X over the set T
        for p in X:
            distances = []
            for t in self.T:
                distances.append(Utility.lp_distance(p, t[:-1], self.distance_lp))
            distances = np.array(distances)
            k_nearest_neighbors = np.take(self.T, np.argsort(distances)[:self.k], axis=0)
            # predict the label of the point in X by majority voting
            # by counting the number of red and blue points in the k nearest neighbors
            # and assign the label of the majority
            y[count] = np.argmax(np.bincount(k_nearest_neighbors[:, -1].astype(int).tolist()))
            count += 1
        return y


