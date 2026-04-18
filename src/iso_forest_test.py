import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

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

sensor_cols = ["sensor_1", "sensor_2", "sensor_3", "sensor_4",
            "sensor_5", "sensor_6", "sensor_7", "sensor_8",
            "sensor_9", "sensor_10","sensor_11", "sensor_12",
            "sensor_13", "sensor_14", "sensor_15", "sensor_16",
            "sensor_17", "sensor_18", "sensor_19", "sensor_20", "sensor_21"]

#Change file path, according your own path
train_path = r"C:\Programmieren\anomalie_erkennung_projekt\data\train\train_FD001.txt"
test_path = r"C:\Programmieren\anomalie_erkennung_projekt\data\test\test_FD001.txt"

train_df = read_data(train_path)
test_df = read_data(test_path)

threshold = 0.01
# useful sensors whose standard deviation is less than the threshold, which means they are constant and produce noise for model
useful_sensors = [col for col in sensor_cols if train_df[col].std() > threshold]

#calculate remaining useful life (RUL)
max_cycles = train_df.groupby("unit_id")["cycles"].max().reset_index()
max_cycles.columns = ["unit_id", "max_cycle"]
train_df = train_df.merge(max_cycles, on="unit_id")
train_df["RUL"] = train_df["max_cycle"] - train_df["cycles"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(train_df[useful_sensors])

iso_forest = IsolationForest(n_estimators=100, contamination=0.05, max_samples=256, random_state=42)
iso_forest.fit(X_scaled)
train_df["anomaly_score"] = iso_forest.decision_function(X_scaled)
train_df["anomaly"] = iso_forest.predict(X_scaled)

# rolling window mean logic to reduce noise and better estimate decline better
window = 15
train_df["score_rolling"] = train_df.groupby("unit_id")["anomaly_score"].transform(lambda x: x.rolling(window).mean())

# Filter healthy cycles to later calculate mean and variance for threshold
healthy_filter = train_df["cycles"] <= train_df["max_cycle"] * 0.3
healthy_cycles = train_df[healthy_filter]

healthy_cycles_mean = healthy_cycles.groupby("unit_id")["score_rolling"].mean()
healthy_cycles_mean = healthy_cycles_mean.reset_index()
healthy_cycles_mean.columns = ["unit_id", "score_mean"]
train_df = train_df.merge(healthy_cycles_mean, on="unit_id")

healthy_cycles_std = healthy_cycles.groupby("unit_id")["score_rolling"].std()
healthy_cycles_std = healthy_cycles_std.reset_index()
healthy_cycles_std.columns = ["unit_id", "score_std"]
train_df = train_df.merge(healthy_cycles_std, on="unit_id")

score_normalized = (train_df["score_rolling"] - train_df["score_mean"]) / train_df["score_std"]
train_df["score_normalized"] = score_normalized

anomaly_flag = train_df["score_normalized"] < -2
train_df["anomaly_flag"] = anomaly_flag

first_anomaly = (
    train_df[train_df["anomaly_flag"] == 1]
    .groupby("unit_id")["RUL"]
    .max()  # max RUL beim ersten Alarm = wie früh wird erkannt
)

print(first_anomaly.describe())


fig, axes = plt.subplots(5, 2, figsize=(14, 18), sharex=False)

for i, unit_id in enumerate(range(1, 6)):
    unit = train_df[train_df["unit_id"] == unit_id].sort_values("cycles")
    axes[i][0].plot(unit["cycles"], unit["RUL"])
    axes[i][0].set_ylabel(f"Unit {unit_id} RUL")
    axes[i][1].plot(unit["cycles"], unit["anomaly_score"])
    axes[i][1].set_ylabel("Anomaly Score")

    plt.tight_layout()
plt.show()
