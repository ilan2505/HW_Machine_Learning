from typing import Union
import numpy as np
class Circle:
    def __init__(self, center: np.ndarray, outer: np.ndarray, color: {1, -1}):
        self.center = center
        self.outer = outer
        self.radius = np.linalg.norm(center - outer)
        self.color = color

    def __distance_from_point(self, point: np.ndarray) -> np.ndarray:
        # Return the distance from this line to a point
        return np.linalg.norm(point - self.center, axis=1) - self.radius

    def predict(self, point: np.ndarray) -> Union[int, np.ndarray]:  # Return the class of the point
        distances = self.__distance_from_point(point)
        if self.color == 1:
            return np.where(distances > 0, 1, -1)
        else:
            return np.where(distances < 0, 1, -1)

    def __str__(self):
        return "Circle: " + str(self.center) + " radius: " + str(self.radius)