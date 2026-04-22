import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def fit_pipeline(df, scaled_values, n_estimators, contamination, max_samples, window, z_threshold):
    run_df = df.copy()

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples=max_samples,
        random_state=42,
    )
    model.fit(scaled_values)
    run_df["anomaly_score"] = model.decision_function(scaled_values)

    run_df["score_rolling"] = run_df.groupby("unit_id")["anomaly_score"].transform(
        lambda x: x.rolling(window).mean()
    )

    healthy_filter = run_df["cycles"] <= run_df["max_cycle"] * 0.3
    healthy_cycles = run_df[healthy_filter]

    score_mean = healthy_cycles.groupby("unit_id")["score_rolling"].mean().reset_index()
    score_mean.columns = ["unit_id", "score_mean"]
    score_std = healthy_cycles.groupby("unit_id")["score_rolling"].std().reset_index()
    score_std.columns = ["unit_id", "score_std"]

    run_df = run_df.merge(score_mean, on="unit_id")
    run_df = run_df.merge(score_std, on="unit_id")

    run_df["score_normalized"] = (
        (run_df["score_rolling"] - run_df["score_mean"]) / run_df["score_std"]
    ).replace([np.inf, -np.inf], np.nan)

    run_df["anomaly_flag"] = (run_df["score_normalized"] < z_threshold).fillna(False)

    return run_df, model, score_mean, score_std


def predict_pipeline(df, scaled_values, model, score_mean, score_std, window, z_threshold):
    run_df = df.copy()

    run_df["anomaly_score"] = model.decision_function(scaled_values)

    run_df["score_rolling"] = run_df.groupby("unit_id")["anomaly_score"].transform(
        lambda x: x.rolling(window).mean()
    )

    run_df = run_df.merge(score_mean, on="unit_id")
    run_df = run_df.merge(score_std, on="unit_id")

    run_df["score_normalized"] = (
        (run_df["score_rolling"] - run_df["score_mean"]) / run_df["score_std"]
    ).replace([np.inf, -np.inf], np.nan)

    run_df["anomaly_flag"] = (run_df["score_normalized"] < z_threshold).fillna(False)

    return run_df
