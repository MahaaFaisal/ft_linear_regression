# import pandas as pd
import numpy as np
from load_data import load_train_data

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
    try:
        x_train, y_train = load_train_data("data.csv")
        w, b = 0, 0
        rate = 1e-4
        m = len(x_train)
        iterations = 1000000
        cost = 80
        
        for i in range(iterations):
            f_wb = calculate_f_wb(x_train, m, w, b)
            dj_w, dj_b = calculate_gradient(f_wb, x_train, y_train, m)
            w = w - (rate * dj_w)
            b = b - (rate * dj_b)
            f_wb = calculate_f_wb(x_train, m, w, b)
            prev_cost = cost
            cost = calculate_cost(f_wb, y_train, m)
            if prev_cost - cost < 1e-6:
                break
            print(f"{i}th iteration: cost = {cost}")
        np.savez("model_params.npz", w=w, b=b)

    # repeat above steps until we reach convergence (or just choose a big number of iterations??)

    except Exception as e:
        print(f"{type(e).__name__}: {e}")    



if __name__ == "__main__":
    main()