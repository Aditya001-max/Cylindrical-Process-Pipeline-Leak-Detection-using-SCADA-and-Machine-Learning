"""
Pipeline Leak Detection Model Training & Evaluation
===================================================
Author: Aditya001-max (kaushikadityaraj001@gmail.com)
Project: Cylindrical Process Pipeline Leak Detection using SCADA and Machine Learning

This module trains a weighted Random Forest classifier on hydraulic SCADA sensor features
to detect pipeline leaks under class imbalance (~4:1 to 12:1 imbalance). It validates performance
with 5-fold Stratified Cross-Validation, optimizes decision threshold (0.30 for high recall),
extracts feature importance rankings, and exports the production model artifact (model.pkl).
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

FEATURES = ["pressure_diff", "flow_diff", "pressure_roll_mean", "flow_roll_std"]
TARGET = "leak"


def prepare_dataset(csv_path: str = "pipeline_data_realistic.csv") -> pd.DataFrame:
    """
    Loads telemetry dataset and computes rolling hydraulic features if missing.
    """
    if not os.path.exists(csv_path):
        from generate_data import generate_pipeline_telemetry
        print(f"[INFO] Telemetry dataset '{csv_path}' not found. Generating synthetic dataset...")
        return generate_pipeline_telemetry(output_processed_csv=csv_path)

    df = pd.read_csv(csv_path)

    # Ensure required features exist
    if not all(col in df.columns for col in FEATURES):
        if "pressure_in" in df.columns and "pressure_out" in df.columns:
            df["pressure_diff"] = df["pressure_in"] - df["pressure_out"]
            df["flow_diff"] = df["flow_in"] - df["flow_out"]
            df["pressure_roll_mean"] = df["pressure_diff"].rolling(5).mean()
            df["flow_roll_std"] = df["flow_diff"].rolling(5).std()
            df = df.dropna().reset_index(drop=True)
        else:
            raise ValueError(f"Dataset missing required hydraulic columns: {FEATURES}")

    return df


def train_and_evaluate(
    data_path: str = "pipeline_data_realistic.csv",
    model_output_path: str = "model.pkl",
    test_size: float = 0.20,
    random_state: int = 42,
    threshold: float = 0.30,
    n_splits: int = 5
):
    """
    Executes full model training, 5-fold cross-validation, threshold evaluation, and model export.
    """
    df = prepare_dataset(data_path)
    X = df[FEATURES]
    y = df[TARGET].astype(int)

    leak_count = (y == 1).sum()
    normal_count = (y == 0).sum()
    imbalance_ratio = normal_count / max(1, leak_count)
    print(f"\n==================================================")
    print(f" PIPELINE SCADA LEAK DETECTION MODEL TRAINING")
    print(f"==================================================")
    print(f"Dataset Shape   : {df.shape[0]} samples × {df.shape[1]} columns")
    print(f"Class Balance   : Normal={normal_count} ({(1 - y.mean())*100:.1f}%), Leak={leak_count} ({y.mean()*100:.1f}%)")
    print(f"Imbalance Ratio : {imbalance_ratio:.2f}:1")

    # 1. Stratified 5-Fold Cross Validation
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rf_cv = RandomForestClassifier(class_weight="balanced", random_state=random_state)
    cv_auc_scores = cross_val_score(rf_cv, X, y, cv=cv, scoring="roc_auc")

    print(f"\n[5-Fold Stratified Cross-Validation]")
    for i, score in enumerate(cv_auc_scores, 1):
        print(f"  Fold {i} ROC-AUC: {score:.4f}")
    print(f"  Mean ROC-AUC   : {cv_auc_scores.mean():.4f} (+/- {cv_auc_scores.std():.4f})")

    # 2. Train / Test Split for Threshold Optimization & Test Evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=random_state
    )
    model.fit(X_train, y_train)

    # Predictions & Probabilities
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred_default = model.predict(X_test)
    y_pred_tuned = (y_prob >= threshold).astype(int)

    test_auc = roc_auc_score(y_test, y_prob)

    print(f"\n[Held-Out Test Set Performance (test_size={test_size})]")
    print(f"  Test ROC-AUC   : {test_auc:.4f}")

    print(f"\n  -- Default Threshold (0.50) --")
    print(f"  Precision (Leak): {precision_score(y_test, y_pred_default, zero_division=0):.4f}")
    print(f"  Recall (Leak)   : {recall_score(y_test, y_pred_default, zero_division=0):.4f}")
    print(f"  F1 Score (Leak) : {f1_score(y_test, y_pred_default, zero_division=0):.4f}")

    print(f"\n  -- Optimized Alarm Threshold ({threshold:.2f}) --")
    print(f"  Precision (Leak): {precision_score(y_test, y_pred_tuned, zero_division=0):.4f}")
    print(f"  Recall (Leak)   : {recall_score(y_test, y_pred_tuned, zero_division=0):.4f}")
    print(f"  F1 Score (Leak) : {f1_score(y_test, y_pred_tuned, zero_division=0):.4f}")

    cm = confusion_matrix(y_test, y_pred_tuned)
    print(f"\n  Confusion Matrix @ threshold={threshold:.2f}:")
    print(f"  [TN: {cm[0, 0]:>4} | FP: {cm[0, 1]:>4}]")
    print(f"  [FN: {cm[1, 0]:>4} | TP: {cm[1, 1]:>4}]")

    # 3. Feature Importance Analysis
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print(f"\n[Feature Importance Ranking]")
    for rank, idx in enumerate(sorted_idx, 1):
        print(f"  {rank}. {FEATURES[idx]:<20}: {importances[idx]*100:>5.2f}%")

    # 4. Export Artifact
    joblib.dump(model, model_output_path)
    print(f"\n[OK] Model successfully serialized and exported to '{model_output_path}'\n")

    return {
        "model": model,
        "cv_mean_auc": cv_auc_scores.mean(),
        "test_auc": test_auc,
        "importances": dict(zip(FEATURES, importances))
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train pipeline leak detection model.")
    parser.add_argument("--data", type=str, default="pipeline_data_realistic.csv", help="Path to input dataset")
    parser.add_argument("--output", type=str, default="model.pkl", help="Path to save model artifact")
    parser.add_argument("--threshold", type=float, default=0.30, help="Operational alarm probability threshold")
    parser.add_argument("--cv-splits", type=int, default=5, help="Number of cross-validation folds")
    args = parser.parse_args()

    train_and_evaluate(
        data_path=args.data,
        model_output_path=args.output,
        threshold=args.threshold,
        n_splits=args.cv_splits
    )
