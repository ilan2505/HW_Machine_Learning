import numpy as np

FILE_NAME = "circle_separator.txt"


def read_data(filename: str) -> np.ndarray:
    with open(filename) as f:
        lines = f.readlines()
    data = np.zeros((len(lines), 3))
    for i in range(len(lines)):
        data[i] = np.array([float(x) for x in lines[i].split()])
    return data


# split the data into train and test in ratio 1:1
def random_train_test_split(data: np.ndarray) -> (np.ndarray, np.ndarray):
    shuffled_data = np.random.permutation(data)
    train_data = shuffled_data[0: int(len(data) * 0.5)]
    test_data = shuffled_data[int(len(data) * 0.5):]
    return train_data, test_data
