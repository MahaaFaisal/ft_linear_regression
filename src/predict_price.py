import numpy as np


def load_parameters(path: str):
    try:
        data = np.load(path)
        w = data['w']
        b = data['b']
        x_mu = data['x_mu']
        x_std = data['x_std']
        y_mu = data['y_mu']
        y_std = data['y_std']
        return w, b, x_mu, x_std, y_mu, y_std
    except Exception:
        print("failed to load parameters, calculating with 0")
        return 0, 0, 0, 0, 0, 0


def get_user_input():
    while True:
        try:
            miliage = input("How many miles did your car go?? ")
            miliage = float(miliage)
            break
        except Exception:
            print("Invalid number, try again or press ctrl+D to exit")
    return miliage


def predict_price(x: float, w: float, b: float, x_mu, x_std, y_mu, y_std):
    scaled_x = (x - x_mu) / x_std
    scaled_pred = w * scaled_x + b
    return scaled_pred * y_std + y_mu


def main():
    try:
        w, b, x_mu, x_std, y_mu, y_std = load_parameters("../model_params.npz")
        # assert isinstance(w, float) and isinstance(b, float), "w and b should be float"
        miliage = get_user_input()
        print(f"the estimated price is:\
              {predict_price(miliage, w, b, x_mu, x_std, y_mu, y_std)}")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
    except KeyboardInterrupt:
        print("\nProgram interrupted")


if __name__ == "__main__":
    main()
