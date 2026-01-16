
def load_parameters(path: str):
    pass

def get_user_input():
    while True:
        try:
            miliage = input("How many miles did your car go?? ")
            miliage = float(miliage)
            break
        except:
            print("Invalid number, try again or press ctrl+D to exit")
    return miliage


def predict_price(miliage: float, w: float, b: float):
    return w * miliage + b

def main():
    w, b = load_parameters("parameters.txt")
    miliage = get_user_input()
    print(f"the estimated price is: {predict_price(miliage, w, b)}")


if __name__ == "__main__":
    main()