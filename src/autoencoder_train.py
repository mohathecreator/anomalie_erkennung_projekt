import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from itertools import product
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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
train_df[useful_sensors] = X_scaled

def create_sequences(df, window_size, features):
    x = []
    labels = []

    for unit_id in df["unit_id"].unique():
        unit = df[df["unit_id"] == unit_id].sort_values("cycles")
        unit_values = unit[features].values

        for i in range(len(unit_values) - window_size +1):
            window = unit_values[i:i + window_size]
            x.append(window)
            labels.append((unit_id, unit["cycles"].values[i + window_size -1]))

    return np.array(x), np.array(labels)

healthy_filter = train_df["cycles"] <= train_df["max_cycle"] * 0.3
healthy_df = train_df[healthy_filter]

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        hidden = hidden.permute(1, 0, 2)
        hidden = hidden.repeat(1, self.seq_len, 1)
        output, _ = self.decoder(hidden)
        return self.output_layer(output)


def run_pipeline(df, healthy_subset, features, window_size, hidden_size, lr, epochs, threshold):
    torch.manual_seed(42)

    model = LSTMAutoencoder(input_size=len(features), hidden_size=hidden_size, seq_len=window_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_sequences, _ = create_sequences(healthy_subset, window_size, features)
    if len(X_sequences) == 0:
        return None

    X_tensor = torch.FloatTensor(X_sequences)
    dataset = TensorDataset(X_tensor, X_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for _ in range(epochs):
        for x_batch, y_batch in loader:
            output = model(x_batch)
            loss = criterion(output, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    all_sequences, labels = create_sequences(df, window_size, features)
    if len(all_sequences) == 0:
        return None

    all_tensor = torch.FloatTensor(all_sequences)
    with torch.no_grad():
        reconstructed = model(all_tensor)
        reconstruction_error = torch.mean((all_tensor - reconstructed) ** 2, dim=(1, 2))

    run_results = pd.DataFrame(labels, columns=["unit_id", "cycle"])
    run_results["reconstruction_error"] = reconstruction_error.numpy()
    run_results = run_results.merge(
        df[["unit_id", "cycles", "RUL"]],
        left_on=["unit_id", "cycle"],
        right_on=["unit_id", "cycles"],
    )
    run_results["anomaly_flag"] = run_results["reconstruction_error"] > threshold
    return run_results


param_grid = {
    "window_size": [20, 30],
    "hidden_size": [16, 32],
    "lr": [0.001, 0.0005],
    "epochs": [30, 50],
    "threshold": [0.2, 0.3, 0.4],
}

search_results = []
best_result = None
best_results_df = None

for window_size, hidden_size, lr, epochs, threshold in product(
    param_grid["window_size"],
    param_grid["hidden_size"],
    param_grid["lr"],
    param_grid["epochs"],
    param_grid["threshold"],
):
    candidate_df = run_pipeline(
        train_df,
        healthy_df,
        useful_sensors,
        window_size,
        hidden_size,
        lr,
        epochs,
        threshold,
    )
    if candidate_df is None:
        continue

    y_true = (candidate_df["RUL"] <= 30).astype(int)
    y_pred = candidate_df["anomaly_flag"].astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    row = {
        "window_size": window_size,
        "hidden_size": hidden_size,
        "lr": lr,
        "epochs": epochs,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    search_results.append(row)

    if best_result is None or (f1, recall, precision) > (
        best_result["f1"],
        best_result["recall"],
        best_result["precision"],
    ):
        best_result = row
        best_results_df = candidate_df

results_overview_df = pd.DataFrame(search_results).sort_values(
    by=["f1", "recall", "precision"], ascending=False
)
print("Top 10 Gridsearch-Ergebnisse:")
print(results_overview_df.head(10).to_string(index=False))

print("\nBeste Parameter:")
print(
    f"window_size={best_result['window_size']}, "
    f"hidden_size={best_result['hidden_size']}, "
    f"lr={best_result['lr']}, "
    f"epochs={best_result['epochs']}, "
    f"threshold={best_result['threshold']}"
)

results_df = best_results_df

first_anomaly = (
    results_df[results_df["anomaly_flag"] == 1]
    .groupby("unit_id")["RUL"]
    .max()  # max RUL beim ersten Alarm = wie früh wird erkannt
)

print(first_anomaly.describe())

y_true = (results_df["RUL"] <= 30).astype(int)
y_pred = (results_df["anomaly_flag"]).astype(int)

print(f"Precision: {precision_score(y_true, y_pred)}, Recall: {recall_score(y_true, y_pred)}")
print(f"F1: {f1_score(y_true, y_pred)}")

fig, axes = plt.subplots(5, 2, figsize=(14, 18), sharex=False)

for i, unit_id in enumerate(range(1, 6)):
    unit = results_df[results_df["unit_id"] == unit_id].sort_values("cycles")
    axes[i][0].plot(unit["cycles"], unit["RUL"])
    axes[i][0].set_ylabel(f"Unit {unit_id} RUL")
    axes[i][1].plot(unit["cycles"], unit["reconstruction_error"])
    axes[i][1].set_ylabel("Reconstruction Error")

    plt.tight_layout()
plt.show()

first_anomaly.hist(bins=20)
plt.xlabel("RUL beim ersten Alarm")
plt.show()
