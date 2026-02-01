import pandas as pd


def check_path(path: str) -> None:
    if not isinstance(path, str):
        raise TypeError("file path should be a string")
    if not path.endswith(".csv"):
        raise ValueError("file extenion should be .csv")


def load_file(path: str) -> pd.DataFrame:
    """ a fuunction that takes a csv file and\
loads it to dataframe
    """
    check_path(path)
    df = pd.read_csv(path)
    print(f"Loading dataset of dimensions {df.shape}")
    return df


def load_train_data(path: str):
    df = load_file(path)
    x = df["km"].to_numpy()
    y = df["price"].to_numpy()
    return x, y