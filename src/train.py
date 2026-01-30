# import pandas as pd
import numpy as np
from load_data import load_train_data
from plot import plot_regression
import matplotlib.pyplot as plt


def calculate_f_wb(x: np.ndarray, w: float, b: float) -> np.ndarray:
    return w * x + b


def calculate_cost(f_wb: np.ndarray, y_train: np.ndarray, m: int) -> float:
    cost = np.sum((f_wb - y_train) ** 2) / (2 * m)

    return cost


def calculate_gradient(f_wb: np.ndarray, x_train: np.ndarray, y_train: np.ndarray, m: int) -> tuple:
    dj_w = 0
    dj_b = 0

    dj_w = np.sum(x_train * (f_wb - y_train))/m
    dj_b = np.sum(f_wb - y_train)/m

    return dj_w, dj_b


def gradient_descent(x_train: np.ndarray, y_train: np.ndarray) -> tuple:
    w, b = 0, 0
    rate = 1
    m = len(x_train)
    iterations = 200000

    f_wb = calculate_f_wb(x_train, w, b)
    cost = calculate_cost(f_wb, y_train, m)
    for i in range(iterations):
        dj_w, dj_b = calculate_gradient(f_wb, x_train, y_train, m)
        w = w - (rate * dj_w)
        b = b - (rate * dj_b)
        f_wb = calculate_f_wb(x_train, w, b)
        prev_cost = cost
        cost = calculate_cost(f_wb, y_train, m)
        if abs(prev_cost - cost) < 1e-6:
            break
        print(f"{i}th iteration: cost = {cost}")
    return w, b


def main():
    try:
        x_train, y_train, x_mu, x_std, y_mu, y_std = load_train_data("../data.csv")
        print(x_train, y_train)
        w, b = gradient_descent(x_train, y_train)
        plot_regression(x_train, y_train, w, b, x_mu, x_std, y_mu, y_std)
        np.savez("../model_params.npz", w=w, b=b, x_mu=x_mu, x_std=x_std, y_mu=y_mu, y_std=y_std)

    except Exception as e:
        print(f"{type(e).__name__}: {e}")
    except KeyboardInterrupt:
        print("\nProgram interrupted")
        plt.close()


if __name__ == "__main__":
    main()
