import numpy as np
from load_data import load_train_data
from plot import plot_regression
import matplotlib.pyplot as plt

class LinearRegression():
    def __init__(self):
        self.w = 0
        self.b = 0
        self.x_mu = self.y_mu = 0
        self.x_std = self.y_std = 0
        self.is_trained = False
    def _calculate_gradient(self, f_wb, x_train, y_train, m):
        dj_w = 0
        dj_b = 0

        dj_w = np.sum(x_train * (f_wb - y_train))/m
        dj_b = np.sum(f_wb - y_train)/m

        return dj_w, dj_b

    def train(self, x_train, y_train, rate = 0.1, iterations = 200000):
        m = len(x_train)
        x_train = self.scale_x(x_train)
        y_train = self.scale_y(y_train)

        f_wb = self.w * x_train + self.b
        cost = np.sum((f_wb - y_train) ** 2) / (2 * m)
        for i in range(iterations):
            dj_w, dj_b = self._calculate_gradient(f_wb, x_train, y_train, m)
            self.w = self.w - (rate * dj_w)
            self.b = self.b - (rate * dj_b)
            f_wb = self.w * x_train + self.b
            prev_cost = cost
            cost = np.sum((f_wb - y_train) ** 2) / (2 * m)
            if abs(prev_cost - cost) < 1e-6:
                break
            print(f"{i}th iteration: cost = {cost}")
        self.is_trained = True
        return self.w, self.b

    def load_parameters(self, path):
        try:
            data = np.load(path)
            self.w = data['w']
            self.b = data['b']
            self.x_mu = data['x_mu']
            self.x_std = data['x_std']
            self.y_mu = data['y_mu']
            self.y_std = data['y_std']
            self.is_trained = True

        except Exception:
            print("failed to load parameters, using untrained model")
            self.w = 0
            self.b = 0
            self.x_mu = 0
            self.x_std = 1
            self.y_mu = 0
            self.y_std = 1

    def scale_x(self, x):
        if (isinstance(x, np.ndarray)):
            self.x_mu = x.mean()
            self.x_std = x.std(ddof=0)
        return (x - self.x_mu) / self.x_std

    def unscale_x(self, x_scaled):
        return x_scaled * self.x_std + self.x_mu

    def scale_y(self, y):
        self.y_mu = y.mean()
        self.y_std = y.std(ddof=0)
        return (y - self.y_mu) / self.y_std

    def unscale_y(self, y_scaled):
        return y_scaled * self.y_std + self.y_mu
    
    def predict(self, x_train):
        scaled_y = self.w * self.scale_x(x_train) + self.b
        return self.unscale_y(scaled_y)