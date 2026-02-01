from linear_regression import LinearRegression
from load_data import load_train_data
import numpy as np


def evaluate(x_train, y_train, y_pred, model):
    ss_res = np.sum((y_train - y_pred)**2)
    ss_total = np.sum((y_train - model.y_mu)**2)
    r_squared = 1 - (ss_res / ss_total)
    return r_squared


def main():
    try:
        x_train, y_train = load_train_data("../data.csv")
        model = LinearRegression()
        model.load_parameters("../model_params.npz")
        y_pred = model.predict(x_train)

        r_squared = evaluate(x_train, y_train, y_pred, model)
        print("r_squared = ", r_squared)

    except Exception as e:
        print(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
