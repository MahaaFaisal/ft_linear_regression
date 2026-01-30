import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

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

def z_score_scale(x: pd.Series) -> pd.Series:
    mu = x.mean()
    std = x.std(ddof=0)
    z_score_scaled = (x - mu) / std
    return z_score_scaled, mu, std


def load_train_data(path: str):
    df = load_file(path)
    x = df["km"]
    x_scaled, x_mu, x_std = z_score_scale(x)
    y = df["price"]
    y_scaled, y_mu, y_std = z_score_scale(y)
    return x_scaled.to_numpy(), y_scaled.to_numpy(), x_mu, x_std, y_mu, y_std
