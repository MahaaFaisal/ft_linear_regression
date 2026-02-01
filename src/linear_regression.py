import numpy as np
import matplotlib.pyplot as plt


class LinearRegression():
    def __init__(self):
        self.theta1 = 0
        self.theta0 = 0
        self.x_mu = self.y_mu = 0
        self.x_std = self.y_std = 1
        self.is_trained = False


    def _calculate_gradient(self, f_x, x_train, y_train, m):
        dj_theta1 = 0
        dj_theta0 = 0

        dj_theta1 = np.sum(x_train * (f_x - y_train))/m
        dj_theta0 = np.sum(f_x - y_train)/m

        return dj_theta1, dj_theta0


    def train(self, x_train, y_train, rate = 0.1, iterations = 200000):
        m = len(x_train)
        x_train = self._scale_x(x_train)
        y_train = self._scale_y(y_train)

        f_x = self.theta1 * x_train + self.theta0
        cost = np.sum((f_x - y_train) ** 2) / (2 * m)

        for i in range(iterations):
            dj_theta1, dj_theta0 = self._calculate_gradient(f_x, x_train, y_train, m)
            
            tmp_theta1 = rate * dj_theta1
            tmp_theta0 = rate * dj_theta0
            
            self.theta1 = self.theta1 - tmp_theta1
            self.theta0 = self.theta0 - tmp_theta0
            f_x = self.theta1 * x_train + self.theta0
            
            prev_cost = cost
            cost = np.sum((f_x - y_train) ** 2) / (2 * m)
            if abs(prev_cost - cost) < 1e-6:
                break
            print(f"{i}th iteration: cost = {cost}")
        
        self.is_trained = True
        return self.theta1, self.theta0

    
    def predict(self, x_train):
        scaled_y = self.theta1 * self._scale_x(x_train) + self.theta0
        return self._unscale_y(scaled_y)


    def _scale_x(self, x):
        if (isinstance(x, np.ndarray)):
            self.x_mu = x.mean()
            self.x_std = x.std(ddof=0)
        return (x - self.x_mu) / self.x_std

    def _unscale_x(self, x_scaled):
        return x_scaled * self.x_std + self.x_mu

    def _scale_y(self, y):
        self.y_mu = y.mean()
        self.y_std = y.std(ddof=0)
        return (y - self.y_mu) / self.y_std

    def _unscale_y(self, y_scaled):
        return y_scaled * self.y_std + self.y_mu


    def load_parameters(self, path):
        try:
            data = np.load(path)
            self.theta1 = data['theta1']
            self.theta0 = data['theta0']
            self.x_mu = data['x_mu']
            self.x_std = data['x_std']
            self.y_mu = data['y_mu']
            self.y_std = data['y_std']
            self.is_trained = True

        except Exception:
            print("failed to load parameters, using untrained model")


    def save_npz(self, path):
        np.savez(path,
            theta1=self.theta1,
            theta0=self.theta0,
            x_mu=self.x_mu,
            x_std=self.x_std,
            y_mu=self.y_mu,
            y_std=self.y_std)