import matplotlib.pyplot as plt
import numpy as np
from predict_price import predict_price


def predict_y(x_train: np.ndarray, w: float, b: float):
    y_predict = [predict_price(x, w, b) for x in x_train]
    return y_predict


def plot_data(x_train: np.ndarray, y_train: np.ndarray,):
    plt.scatter(x_train * 1000, y_train * 1000, marker='x', color="red")
    pass


def plot_line(x_train: np.ndarray, y_predict: np.ndarray):
    plt.plot(x_train * 1000, y_predict, color="black", linewidth=0.5)
    pass


def plot_regression(x_train: np.ndarray, y_train: np.ndarray,
                    w: float, b: float) -> None:
    plt.title("Relationship Between Car Mileage and Price")
    plt.xlabel("miliage")
    plt.ylabel("price in KM")
    plot_data(x_train, y_train)
    y_predict = predict_y(x_train, w, b)
    plot_line(x_train, y_predict)
    plt.show()
