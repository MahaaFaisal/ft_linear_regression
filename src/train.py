# import pandas as pd
import numpy as np
from load_data import load_train_data
from plot import plot_regression
import matplotlib.pyplot as plt
from tqdm import tqdm


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
    dj_w = 0
    dj_b = 0

    for i in range(m):
        dj_w = dj_w + (x_train[i] * (f_wb[i] - y_train[i]))
        dj_b = dj_b + (f_wb[i] - y_train[i])
    dj_w = dj_w / m
    dj_b = dj_b / m
    return dj_w, dj_b


def gradient_descent(x_train: np.ndarray, y_train: np.ndarray) -> tuple:
    w, b = 0, 0
    rate = 1e-4
    m = len(x_train)
    iterations = 200000
    cost = 80

    f_wb = calculate_f_wb(x_train, m, w, b)
    for i in range(iterations):
        dj_w, dj_b = calculate_gradient(f_wb, x_train, y_train, m)
        w = w - (rate * dj_w)
        b = b - (rate * dj_b)
        f_wb = calculate_f_wb(x_train, m, w, b)
        prev_cost = cost
        cost = calculate_cost(f_wb, y_train, m)
        if prev_cost - cost < 1e-6:
            break
        print(f"{i}th iteration: cost = {cost}")
    return w, b


def main():
    try:
        x_train, y_train = load_train_data("../data.csv")
        w, b = gradient_descent(x_train, y_train)
        plot_regression(x_train, y_train, w, b)
        np.savez("../model_params.npz", w=w, b=b)

    except Exception as e:
        print(f"{type(e).__name__}: {e}")
    except KeyboardInterrupt:
        print("\nProgram interrupted")
        plt.close()


if __name__ == "__main__":
    main()
