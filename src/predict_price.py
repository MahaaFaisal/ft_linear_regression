import numpy as np
from linear_regression import LinearRegression


def get_user_input():
    while True:
        try:
            miliage = input("How many miles did your car go?? ")
            miliage = float(miliage)
            break
        except Exception:
            print("Invalid number, try again or press ctrl+D to exit")
    return miliage


def main():
    try:
        model = LinearRegression()
        model.load_parameters("../model_params.npz")
        # assert isinstance(w, float) and isinstance(b, float), "w and b should be float"
        miliage = get_user_input()
        print(f"the estimated price is: {model.predict(miliage)}")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
    except KeyboardInterrupt:
        print("\nProgram interrupted")


if __name__ == "__main__":
    main()
