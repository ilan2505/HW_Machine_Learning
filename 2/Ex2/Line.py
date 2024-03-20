from typing import Union
import numpy as np
class Line:
    def __init__(self, p1: np.ndarray, p2: np.ndarray, color: {1, -1}):
        self.p1 = p1
        self.p2 = p2
        self.color = color
        self.v = p2 - p1
        self.norm_v = np.linalg.norm(self.v)

    def __distance_from_point(self, point: np.ndarray) -> float:
        # Return the distance from this line to a point
        return np.cross(self.v, point - self.p1) / self.norm_v

    def predict(self, point: np.ndarray) -> Union[int, np.ndarray]:  # Return the class of the point
        distances = self.__distance_from_point(point)
        if self.color == 1:
            return np.where(distances > 0, 1, -1)
        else:
            return np.where(distances < 0, 1, -1)

    def __str__(self):
        return "Line: " + str(self.p1) + " to " + str(self.p2)