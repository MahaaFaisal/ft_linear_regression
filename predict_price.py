import numpy as np


def load_parameters(path: str):
    data = np.load(path)
    w = data['w']
    b = data['b']
    return w, b


def get_user_input():
    while True:
        try:
            miliage = input("How many miles did your car go?? ")
            miliage = float(miliage)
            break
        except:
            print("Invalid number, try again or press ctrl+D to exit")
    return miliage / 1000


def predict_price(miliage: float, w: float, b: float):
    return ((w * miliage) + b) * 1000

def main():
    w, b = load_parameters("model_params.npz")
    miliage = get_user_input()
    print(f"the estimated price is: {predict_price(miliage, w, b)}")


if __name__ == "__main__":
    main()