"""
Automated Smoke & Unit Tests for Pipeline SCADA System
=====================================================
Author: Aditya001-max (kaushikadityaraj001@gmail.com)
Project: Cylindrical Process Pipeline Leak Detection using SCADA and Machine Learning
"""

import os
import joblib
import numpy as np
import pandas as pd
import pytest

from generate_data import generate_pipeline_telemetry
from train_model import prepare_dataset, FEATURES, TARGET


def test_data_generation():
    """Verify synthetic SCADA telemetry generator produces valid dataset."""
    df = generate_pipeline_telemetry(
        n_hours=100,
        n_leaks=3,
        n_false_events=5,
        random_seed=123,
        output_raw_csv=None,
        output_processed_csv=None
    )
    assert len(df) > 0, "Generated dataframe should not be empty"
    for feat in FEATURES:
        assert feat in df.columns, f"Feature {feat} must be in generated dataset"
    assert TARGET in df.columns, "Target column 'leak' must be in dataset"
    assert df[FEATURES].isna().sum().sum() == 0, "No NaNs should exist in feature set"


def test_model_loading_and_inference():
    """Verify production model artifact exists and produces valid probabilities."""
    assert os.path.exists("model.pkl"), "model.pkl must exist in workspace"
    model = joblib.load("model.pkl")

    sample_input = pd.DataFrame([[5.0, 3.0, 5.0, 1.0]], columns=FEATURES)  # Standard nominal values
    probs = model.predict_proba(sample_input)

    assert probs.shape == (1, 2), "Inference probabilities must be shape (1, 2)"
    assert 0.0 <= probs[0, 1] <= 1.0, "Leak probability must be bounded between 0 and 1"


def test_alarm_thresholding():
    """Verify threshold logic: alarm triggered when prob >= threshold."""
    model = joblib.load("model.pkl")
    threshold = 0.30

    # Inject significant hydraulic leak signature (pressure drop, flow disparity)
    leak_signature = pd.DataFrame([[-3.0, 25.0, -2.0, 6.0]], columns=FEATURES)
    leak_prob = model.predict_proba(leak_signature)[0, 1]
    is_alarm = int(leak_prob >= threshold)

    assert is_alarm == 1, f"Expected alarm trigger for severe leak signature, got prob={leak_prob}"


def test_dataset_features_integrity():
    """Verify realistic dataset matches expected training schema."""
    df = prepare_dataset("pipeline_data_realistic.csv")
    assert df.shape[0] > 1000, "Dataset should have >1000 operational hours"
    assert set(FEATURES).issubset(df.columns), "Dataset must contain all standard SCADA features"
    assert set(df[TARGET].unique()).issubset({0, 1}), "Leak target must be binary"
