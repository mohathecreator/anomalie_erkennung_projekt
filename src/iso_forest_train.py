import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from itertools import product
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score

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

y_true = (train_df["RUL"] <= 30).astype(int)


def run_pipeline(df, scaled_values, n_estimators, contamination, max_samples, window, z_threshold):
    run_df = df.copy()

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples=max_samples,
        random_state=42,
    )
    model.fit(scaled_values)
    run_df["anomaly_score"] = model.decision_function(scaled_values)
    run_df["anomaly"] = model.predict(scaled_values)

    # rolling window mean logic to reduce noise and better estimate decline better
    run_df["score_rolling"] = run_df.groupby("unit_id")["anomaly_score"].transform(
        lambda x: x.rolling(window).mean()
    )

    # Filter healthy cycles to later calculate mean and variance for threshold
    healthy_filter = run_df["cycles"] <= run_df["max_cycle"] * 0.3
    healthy_cycles = run_df[healthy_filter]

    healthy_cycles_mean = healthy_cycles.groupby("unit_id")["score_rolling"].mean().reset_index()
    healthy_cycles_mean.columns = ["unit_id", "score_mean"]
    run_df = run_df.merge(healthy_cycles_mean, on="unit_id")

    healthy_cycles_std = healthy_cycles.groupby("unit_id")["score_rolling"].std().reset_index()
    healthy_cycles_std.columns = ["unit_id", "score_std"]
    run_df = run_df.merge(healthy_cycles_std, on="unit_id")

    run_df["score_normalized"] = (
        (run_df["score_rolling"] - run_df["score_mean"]) / run_df["score_std"]
    ).replace([np.inf, -np.inf], np.nan)

    run_df["anomaly_flag"] = (run_df["score_normalized"] < z_threshold).fillna(False)
    return run_df


param_grid = {
    "n_estimators": [100, 200],
    "contamination": [0.03, 0.05, 0.08],
    "max_samples": [128, 256],
    "window": [10, 15, 20],
    "z_threshold": [-1.5, -2.0, -2.5],
}

results = []
best_result = None
best_df = None

for n_estimators, contamination, max_samples, window, z_threshold in product(
    param_grid["n_estimators"],
    param_grid["contamination"],
    param_grid["max_samples"],
    param_grid["window"],
    param_grid["z_threshold"],
):
    candidate_df = run_pipeline(
        train_df,
        X_scaled,
        n_estimators,
        contamination,
        max_samples,
        window,
        z_threshold,
    )

    y_pred = candidate_df["anomaly_flag"].astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    row = {
        "n_estimators": n_estimators,
        "contamination": contamination,
        "max_samples": max_samples,
        "window": window,
        "z_threshold": z_threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    results.append(row)

    if best_result is None or (f1, recall, precision) > (
        best_result["f1"],
        best_result["recall"],
        best_result["precision"],
    ):
        best_result = row
        best_df = candidate_df

results_df = pd.DataFrame(results).sort_values(
    by=["f1", "recall", "precision"], ascending=False
)
print("Top 10 Gridsearch-Ergebnisse:")
print(results_df.head(10).to_string(index=False))

print("\nBeste Parameter:")
print(
    f"n_estimators={best_result['n_estimators']}, "
    f"contamination={best_result['contamination']}, "
    f"max_samples={best_result['max_samples']}, "
    f"window={best_result['window']}, "
    f"z_threshold={best_result['z_threshold']}"
)

train_df = best_df

first_anomaly = (
    train_df[train_df["anomaly_flag"] == 1]
    .groupby("unit_id")["RUL"]
    .max()  # max RUL beim ersten Alarm = wie früh wird erkannt
)

print(first_anomaly.describe())
y_pred = (train_df["anomaly_flag"]).astype(int)

print(f"Precision: {precision_score(y_true, y_pred)}, Recall: {recall_score(y_true, y_pred)}")
print(f"F1: {f1_score(y_true, y_pred)}")


fig, axes = plt.subplots(5, 2, figsize=(14, 18), sharex=False)

for i, unit_id in enumerate(range(1, 6)):
    unit = train_df[train_df["unit_id"] == unit_id].sort_values("cycles")
    axes[i][0].plot(unit["cycles"], unit["RUL"])
    axes[i][0].set_ylabel(f"Unit {unit_id} RUL")
    axes[i][1].plot(unit["cycles"], unit["anomaly_score"])
    axes[i][1].set_ylabel("Anomaly Score")

    plt.tight_layout()
plt.show()
