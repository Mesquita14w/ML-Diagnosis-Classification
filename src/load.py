"""
O arquivo loader.py carrega o dataset, remove features inúteis e separa os dados.
"""

import pandas as pd

def load_data(path: str):
    df = pd.read_csv(path)
    df = df.drop(columns=["id", "Unnamed: 32"])

    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]

    return X, y