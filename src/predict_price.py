import numpy as np


def load_parameters(path: str):
    try:
        data = np.load(path)
        w = data['w']
        b = data['b']
        return w, b
    except Exception:
        print("failed to load parameters, calculating with 0")
        return 0, 0


def get_user_input():
    while True:
        try:
            miliage = input("How many miles did your car go?? ")
            miliage = float(miliage)
            break
        except Exception:
            print("Invalid number, try again or press ctrl+D to exit")
    return miliage / 1000


def predict_price(miliage: float, w: float, b: float):
    return ((w * miliage) + b) * 1000


def main():
    try:
        w, b = load_parameters("../model_params.npz")
        assert isinstance(w, float) and isinstance(b, float),"w and b should be float"
        miliage = get_user_input()
        print(f"the estimated price is: {predict_price(miliage, w, b)}")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
    except KeyboardInterrupt:
        print("\nProgram interrupted")


if __name__ == "__main__":
    main()
