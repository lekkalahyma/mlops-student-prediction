import pandas as pd

def load_data(path="data/data.csv"):
    return pd.read_csv(path)

def get_features_target(df):
    X = df[["hours_studied", "marks"]]
    y = df["pass"]
    return X, y