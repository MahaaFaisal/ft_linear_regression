import matplotlib.pyplot as plt
import numpy as np


def plot_data(x_train: np.ndarray, y_train: np.ndarray,):
    plt.scatter(x_train, y_train, marker='x', color="red")
    pass


def plot_line(x_train: np.ndarray, y_predict: np.ndarray):
    plt.plot(x_train, y_predict, color="black", linewidth=0.5)
    pass


def plot_regression(x_train, y_train, model):
    if not model.is_trained:
        raise ValueError("Cannot plot untrained model")
    
    plt.figure(figsize=(10, 6))
    plt.title("Relationship Between Car Mileage and Price")
    plt.xlabel("Mileage (km)")
    plt.ylabel("Price")
    
    plt.scatter(x_train, y_train, marker='x', color='red', 
                label='Training data', alpha=0.7)
    
    y_pred = model.predict(x_train)
    plt.plot(x_train, y_pred, color='black', linewidth=2, 
             label='Regression line')
    
    plt.legend()
    plt.grid(True, alpha=0.9, which='major')
    plt.grid(True, alpha=0.1, which='minor')
    plt.minorticks_on()
    plt.show()
