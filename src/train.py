from load_data import load_train_data
from plot import plot_regression
from linear_regression import LinearRegression


def main():
    try:
        x_train, y_train = load_train_data("../data.csv")
        model = LinearRegression()
        model.train(x_train, y_train)
        plot_regression(x_train, y_train, model)
        model.save_npz("../model_params.npz")

    except Exception as e:
        print(f"{type(e).__name__}: {e}")
    except KeyboardInterrupt:
        print("\nProgram interrupted")


if __name__ == "__main__":
    main()
