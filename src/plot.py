import matplotlib.pyplot as plt
import numpy as np
from predict_price import predict_price


def predict_y(x_train: np.ndarray, w: float, b: float, x_mu, x_std):
    y_pred = w * x_train + b
    return y_pred


def plot_data(x_train: np.ndarray, y_train: np.ndarray,):
    plt.scatter(x_train, y_train, marker='x', color="red")
    pass


def plot_line(x_train: np.ndarray, y_predict: np.ndarray):
    plt.plot(x_train, y_predict, color="black", linewidth=0.5)
    pass


def plot_regression(x_train: np.ndarray, y_train: np.ndarray,
                    w: float, b: float, x_mu, x_std, y_mu, y_std) -> None:
    plt.title("Relationship Between Car Mileage and Price")
    plt.xlabel("miliage")
    plt.ylabel("price in KM")
    x_orig = x_train * x_std + x_mu
    y_orig = y_train * y_std + y_mu
    plot_data(x_orig, y_orig)
    y_predict = predict_y(x_train, w, b, x_mu, x_std)
    y_p_orig = y_predict * y_std + y_mu
    plot_line(x_orig, y_p_orig)
    plt.show()
