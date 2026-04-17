import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
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
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        hidden = hidden.permute(1, 0, 2)
        hidden = hidden.repeat(1, 30, 1)
        output, _ = self.decoder(hidden)
        return self.output_layer(output)

model = LSTMAutoencoder(input_size=14, hidden_size=32)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


window_size = 30

X_sequences, _ = create_sequences(healthy_df, window_size, useful_sensors)
X_tensor = torch.FloatTensor(X_sequences)

dataset = TensorDataset(X_tensor, X_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

epochs = 50

for epoch in range(epochs):
    total_loss = 0
    for x_batch, y_batch in loader:
        output = model(x_batch)
        loss = criterion(output, y_batch)
        optimizer.zero_grad()
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

model.eval()
all_sequences, labels = create_sequences(train_df, window_size, useful_sensors)
all_tensor = torch.FloatTensor(all_sequences)


with torch.no_grad():
    reconstructed = model(all_tensor)
    reconstruction_error = torch.mean((all_tensor - reconstructed) ** 2, dim=(1, 2)) # MSE

reconstruction_error = reconstruction_error.numpy()

results_df = pd.DataFrame(labels, columns=["unit_id", "cycle"])
results_df["reconstruction_error"] = reconstruction_error
results_df = results_df.merge(train_df[["unit_id", "cycles", "RUL"]], left_on=["unit_id", "cycle"], right_on=["unit_id", "cycles"])

anomaly_flag = results_df["reconstruction_error"] > 0.3
results_df["anomaly_flag"] = anomaly_flag

first_anomaly = (
    results_df[results_df["anomaly_flag"] == 1]
    .groupby("unit_id")["RUL"]
    .max()  # max RUL beim ersten Alarm = wie früh wird erkannt
)

print(first_anomaly.describe())

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
