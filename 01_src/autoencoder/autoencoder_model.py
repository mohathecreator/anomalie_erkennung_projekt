import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd


def create_sequences(df, window_size, features):
    x = []
    labels = []
    for unit_id in df["unit_id"].unique():
        unit = df[df["unit_id"] == unit_id].sort_values("cycles")
        unit_values = unit[features].values
        for i in range(len(unit_values) - window_size + 1):
            window = unit_values[i:i + window_size]
            x.append(window)
            labels.append((unit_id, unit["cycles"].values[i + window_size - 1]))
    return np.array(x), np.array(labels)


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


def fit_pipeline(df, healthy_subset, features, window_size, hidden_size, lr, epochs, threshold,
                 patience=5, min_delta=1e-4):
    torch.manual_seed(42)

    model = LSTMAutoencoder(input_size=len(features), hidden_size=hidden_size, seq_len=window_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_sequences, _ = create_sequences(healthy_subset, window_size, features)
    if len(X_sequences) == 0:
        return None, None

    X_tensor = torch.FloatTensor(X_sequences)
    dataset = TensorDataset(X_tensor, X_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    best_loss = float("inf")
    patience_counter = 0

    for _ in range(epochs):
        epoch_loss = 0.0
        for x_batch, y_batch in loader:
            output = model(x_batch)
            loss = criterion(output, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(loader)
        if epoch_loss < best_loss - min_delta:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.eval()
    all_sequences, labels = create_sequences(df, window_size, features)
    all_tensor = torch.FloatTensor(all_sequences)

    with torch.no_grad():
        reconstructed = model(all_tensor)
        reconstruction_error = torch.mean((all_tensor - reconstructed) ** 2, dim=(1, 2))

    results = pd.DataFrame(labels, columns=["unit_id", "cycle"])
    results["reconstruction_error"] = reconstruction_error.numpy()
    results = results.merge(
        df[["unit_id", "cycles", "RUL"]],
        left_on=["unit_id", "cycle"],
        right_on=["unit_id", "cycles"],
    )
    results["anomaly_flag"] = results["reconstruction_error"] > threshold
    return results, model


def predict_pipeline(df, model, features, window_size, threshold):
    model.eval()
    all_sequences, labels = create_sequences(df, window_size, features)
    all_tensor = torch.FloatTensor(all_sequences)

    with torch.no_grad():
        reconstructed = model(all_tensor)
        reconstruction_error = torch.mean((all_tensor - reconstructed) ** 2, dim=(1, 2))

    results = pd.DataFrame(labels, columns=["unit_id", "cycle"])
    results["reconstruction_error"] = reconstruction_error.numpy()
    results["anomaly_flag"] = results["reconstruction_error"] > threshold
    return results