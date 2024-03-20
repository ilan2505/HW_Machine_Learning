from scipy.io.arff import loadarff
import numpy as np

def read_data(filename: str) -> np.ndarray:
    with open(filename) as f:
        lines = f.readlines()
    data = np.zeros((len(lines), 3))
    for i in range(len(lines)):
        data[i] = np.array([float(x) for x in lines[i].split()])
    return data


def read_data_from_csv(filename: str) -> np.ndarray:
    data = loadarff(filename)[0]
    data = np.array([list(x) for x in data])
    data = np.char.decode(data)
    return data

# split the data into train and test in ratio 1:1
def random_train_test_split(data: np.ndarray) -> (np.ndarray, np.ndarray):
    shuffled_data = np.random.permutation(data)
    train_data = shuffled_data[0: int(len(data) * 0.5)]
    test_data = shuffled_data[int(len(data) * 0.5):]
    return train_data, test_data


# calculate distance using Lp norm
def lp_distance(p1: np.ndarray, p2: np.ndarray, p: float = 2) -> float:
    p1 = p1.astype(float)
    p2 = p2.astype(float)
    if p == np.inf:
        return np.max(np.abs(p1 - p2))
    elif p == 0:
        return np.sum(p1 != p2)
    else:
        return np.sum(np.abs(p1 - p2) ** p) ** (1 / p)
