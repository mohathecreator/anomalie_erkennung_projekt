import pandas as pd
from sklearn.ensemble import IsolationForest

def read_data(file_path):
    """Reads data from CSV file and returns a DataFrame."""
    return pd.read_csv(file_path)

data = r"C:\Programmieren\anomalie_erkennung_projekt\anomalie_erkennung_projekt\data\6. Turbofan Engine Degradation Simulation Data Set\CMAPSSData\train_FD001.txt"

print(read_data(data).shape)
