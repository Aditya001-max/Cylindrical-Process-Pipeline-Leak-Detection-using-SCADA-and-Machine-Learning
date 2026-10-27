"""
Resave Model Script (Backward Compatibility Wrapper)
===================================================
Author: Aditya001-max (kaushikadityaraj001@gmail.com)
Project: Cylindrical Process Pipeline Leak Detection using SCADA and Machine Learning

This script maintains backwards compatibility for retraining and serializing the model.
It invokes the unified training pipeline in `train_model.py`.
"""

from train_model import train_and_evaluate

if __name__ == "__main__":
    train_and_evaluate(
        data_path="pipeline_data_realistic.csv",
        model_output_path="model.pkl",
        threshold=0.30
    )
