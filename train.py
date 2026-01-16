import numpy as np

def calculate_f_wb(x: np.ndarray, m: int, w: float, b: float) -> np.ndarray:
    f_wb = np.zeros(m)
    for i in range(m):
      f_wb[i] = (w * x[i]) + b
    return f_wb


def calculate_cost(f_wb: np.ndarray, y_train: np.ndarray, m: int) -> float:
    cost = 0
    for i in range(m):
        cost = cost + (f_wb[i] - y_train[i]) ** 2
    cost = cost / (2 * m)
    return cost


def calculate_gradient(f_wb: np.ndarray, x_train: np.ndarray, y_train: np.ndarray, m: int) -> tuple:
    summation = 0
    dj_w = 0
    dj_b = 0
    
    for i in range(m):
        dj_w = dj_w + (x_train[i] * (f_wb[i] - y_train[i]))
        dj_b = dj_b + (f_wb[i] - y_train[i])
    dj_w = dj_w / m
    dj_b = dj_b / m
    return dj_w, dj_b


def main():
    
    # set w and b to 0
    # learning_rate = 0.1 # arbitrary until I know how to choose an appropriate one
    # load csv
    # separate to x_train, y_train
    # m = df.shape[0]
    # f_wb = calculate_f_wb()
    # cost = calculate_cost (f_wb, y_train, w, b)
    # dj_w, dj_b = calculate_gradient(f_wb_i, x_train, y_train, m)
    # w = w - dj_w
    # b = b - dj_b
    # repeat above steps until we reach convergence (or just choose a big number of iterations??)
    # last step: save final w, b to a file to use in predict.py
    pass
    



if __name__ == "__main__":
    main()