import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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

#print(train_df)
#print(test_df)

features = train_df[["sensor_1", "sensor_2", "sensor_3", "sensor_4",
            "sensor_5", "sensor_6", "sensor_7", "sensor_8",
            "sensor_9", "sensor_10","sensor_11", "sensor_12",
            "sensor_13", "sensor_14", "sensor_15", "sensor_16",
            "sensor_17", "sensor_18", "sensor_19", "sensor_20", "sensor_21"]]

iso_forest = IsolationForest(n_estimators=100, contamination=0.01, max_samples=256)

iso_forest.fit(features)

train_data = train_df.loc[features.index].copy()
train_data["anomaly_score"] = iso_forest.decision_function(features)
train_data["anomaly"] = iso_forest.predict(features)

train_data["anomaly"].value_counts()

# Visualization of the results
plt.figure(figsize=(10, 5))

# Plot normal instances
normal = train_data[train_data['anomaly'] == 1]
plt.scatter(normal.index, normal['anomaly_score'], label='Normal')

# Plot anomalies
anomalies = train_data[train_data['anomaly'] == -1]
plt.scatter(anomalies.index, anomalies['anomaly_score'], label='Anomaly')
plt.xlabel("Instance")
plt.ylabel("Anomaly Score")
plt.legend()
plt.show()

'''
sensor_data1 = train_df[train_df["unit_id"]==1]["sensor_1"]
sensor_data2 = train_df[train_df["unit_id"]==1]["sensor_2"]

plt.plot(sensor_data1)
plt.title("Sensor Daten 1")
plt.xlabel("Time steps")
plt.ylabel("Sensor Value")
plt.show()
'''
