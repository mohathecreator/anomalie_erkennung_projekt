"""Gemeinsames Setup für Praesentations-Notebooks."""
import sys
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).parent.parent
for _subdir in ["01_src", "01_src/autoencoder", "01_src/iso_forest"]:
    _p = str(PROJECT_ROOT / _subdir)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_utils import read_data, sensor_cols


def load_and_prepare():
    """Daten laden, RUL berechnen, Sensoren skalieren."""
    train_df = read_data(PROJECT_ROOT / "02_data" / "train" / "train_FD001.txt")
    test_df  = read_data(PROJECT_ROOT / "02_data" / "test"  / "test_FD001.txt")
    rul_df   = pd.read_csv(
        PROJECT_ROOT / "02_data" / "RUL" / "RUL_FD001.txt",
        header=None, names=["RUL"]
    )
    rul_df["unit_id"] = rul_df.index + 1

    useful_sensors = [c for c in sensor_cols if train_df[c].std() > 0.01]

    max_cycles = train_df.groupby("unit_id")["cycles"].max().reset_index()
    max_cycles.columns = ["unit_id", "max_cycle"]
    train_df = train_df.merge(max_cycles, on="unit_id")
    train_df["RUL"] = train_df["max_cycle"] - train_df["cycles"]

    scaler = StandardScaler()
    train_df[useful_sensors] = scaler.fit_transform(train_df[useful_sensors])
    test_df[useful_sensors]  = scaler.transform(test_df[useful_sensors])

    return train_df, test_df, rul_df, useful_sensors, scaler


def evaluate(y_true, y_pred, label=""):
    if label:
        print(f"{label}:")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1:        {f1_score(y_true, y_pred, zero_division=0):.4f}")


def alarm_analysis(results_df, rul_df):
    alarmed = results_df[results_df["anomaly_flag"] == 1]["unit_id"].unique()
    n_total = rul_df["unit_id"].nunique()
    print(f"Units mit Alarm: {len(alarmed)} / {n_total}")
    print("\nRUL-Verteilung (MIT Alarm):")
    print(rul_df[rul_df["unit_id"].isin(alarmed)]["RUL"].describe().round(1))
    print("\nRUL-Verteilung (OHNE Alarm):")
    print(rul_df[~rul_df["unit_id"].isin(alarmed)]["RUL"].describe().round(1))