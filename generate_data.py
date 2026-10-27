"""
Pipeline SCADA Synthetic Telemetry Generator
============================================
Author: Aditya001-max (kaushikadityaraj001@gmail.com)
Project: Cylindrical Process Pipeline Leak Detection using SCADA and Machine Learning

This module generates a high-fidelity synthetic dataset simulating 2,000+ hours of
cylindrical process pipeline SCADA telemetry. It models normal hydraulic baselines,
realistic sensor noise, 40 gradual leak events with cumulative severity drift, and 50
transient non-leak operational disturbances.
"""

import argparse
import numpy as np
import pandas as pd


def generate_pipeline_telemetry(
    n_hours: int = 2000,
    n_leaks: int = 40,
    n_false_events: int = 50,
    random_seed: int = 42,
    output_raw_csv: str = "pipeline_data.csv",
    output_processed_csv: str = "pipeline_data_realistic.csv"
) -> pd.DataFrame:
    """
    Generates realistic industrial SCADA sensor telemetry and engineers hydraulic features.
    """
    np.random.seed(random_seed)

    # 1. Temporal timeline
    time = pd.date_range(start="2024-01-01", periods=n_hours, freq="h")

    # 2. Baseline hydraulic signals (Normal operations)
    pressure_in = np.random.normal(100.0, 3.0, n_hours)
    flow_in = np.random.normal(500.0, 25.0, n_hours)
    temperature = np.random.normal(30.0, 2.0, n_hours)

    # Baseline outlet hydraulic signals with nominal pressure drop and frictional flow decay
    pressure_out = pressure_in - np.random.normal(5.0, 2.0, n_hours)
    flow_out = flow_in - np.random.normal(3.0, 3.0, n_hours)

    # 3. Ground truth binary leak indicator
    leak = np.zeros(n_hours, dtype=int)

    # 4. Realistic leak injection with progressive severity
    start_min = min(20, n_hours // 10)
    start_max = max(start_min + n_leaks + 1, n_hours - 25)
    valid_range = list(range(start_min, start_max))
    actual_leaks = min(n_leaks, len(valid_range))
    leak_starts = np.random.choice(valid_range, size=actual_leaks, replace=False)
    for start in leak_starts:
        duration = np.random.randint(5, max(6, min(20, n_hours - start)))
        severity = np.random.uniform(2.0, 8.0)  # Gradual hydraulic loss

        for t in range(start, min(start + duration, n_hours)):
            leak[t] = 1
            # Gradual cumulative effect over duration
            factor = (t - start + 1) / duration
            pressure_out[t] -= severity * factor + np.random.normal(0, 1.0)
            flow_out[t] -= (severity * 2.0) * factor + np.random.normal(0, 2.0)

    # 5. Realistic instrument sensor noise
    pressure_out += np.random.normal(0, 1.5, n_hours)
    flow_out += np.random.normal(0, 5.0, n_hours)

    # 6. Transient non-leak disturbances (valve adjustments, pump surges)
    false_events = np.random.choice(range(n_hours), size=n_false_events, replace=False)
    for i in false_events:
        pressure_out[i] -= np.random.uniform(2.0, 5.0)
        flow_out[i] -= np.random.uniform(5.0, 10.0)

    # 7. Assemble raw telemetry dataframe
    df_raw = pd.DataFrame({
        "time": time,
        "pressure_in": pressure_in,
        "pressure_out": pressure_out,
        "flow_in": flow_in,
        "flow_out": flow_out,
        "temperature": temperature,
        "leak": leak
    })

    if output_raw_csv:
        df_raw.to_csv(output_raw_csv, index=False)
        print(f"[OK] Raw telemetry saved -> {output_raw_csv} ({len(df_raw)} records)")

    # 8. Feature engineering: Hydraulic differentials and rolling statistics
    df_feat = df_raw.copy()
    df_feat["pressure_diff"] = df_feat["pressure_in"] - df_feat["pressure_out"]
    df_feat["flow_diff"] = df_feat["flow_in"] - df_feat["flow_out"]
    df_feat["pressure_roll_mean"] = df_feat["pressure_diff"].rolling(5).mean()
    df_feat["flow_roll_std"] = df_feat["flow_diff"].rolling(5).std()

    # Drop warm-up window NaNs
    df_feat = df_feat.dropna().reset_index(drop=True)

    if output_processed_csv:
        df_feat.to_csv(output_processed_csv, index=False)
        print(f"[OK] Processed feature dataset saved -> {output_processed_csv} ({len(df_feat)} records)")

    return df_feat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic SCADA pipeline telemetry.")
    parser.add_argument("--hours", type=int, default=2000, help="Total hours of telemetry simulation")
    parser.add_argument("--leaks", type=int, default=40, help="Number of injected gradual leak events")
    parser.add_argument("--disturbances", type=int, default=50, help="Number of transient non-leak disturbances")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    generate_pipeline_telemetry(
        n_hours=args.hours,
        n_leaks=args.leaks,
        n_false_events=args.disturbances,
        random_seed=args.seed
    )
