import pandas as pd
from sklearn.ensemble import IsolationForest

def read_data(file_path):
    """Reads data from CSV file and returns a DataFrame.""" 
    df = pd.read_csv(file_path, header=None, sep=" ")
    df = df.iloc[:, :26]
    df.columns = columns
    return df

# Names for columns 
columns = ["unit_id", "cycles",
            "op_setting_1", "op_setting_2","op_setting_3",
            "sensor_1", "sensor_2", "sensor_3", "sensor_4",
            "sensor_5", "sensor_6", "sensor_7", "sensor_8",
            "sensor_9", "sensor_10","sensor_11", "sensor_12",
            "sensor_13", "sensor_14", "sensor_15", "sensor_16",
            "sensor_17", "sensor_18", "sensor_19", "sensor_20", "sensor_21"]

#Change file path, according your own path
train_path = r"C:\Programmieren\anomalie_erkennung_projekt\data\train\train_FD001.txt"
test_path = r"C:\Programmieren\anomalie_erkennung_projekt\data\test\test_FD001.txt"

train_df = read_data(train_path)
test_df = read_data(test_path)

print(train_df)
print(test_df)
