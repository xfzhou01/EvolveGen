"""Runtime prediction utility expecting a model file and feature JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import joblib

from feature_utils import ALL_DROP_COLUMNS, prepare_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict solver checking time for a single model.")
    parser.add_argument("model_path", type=Path, help="Path to the trained XGBoost model (.joblib).")
    parser.add_argument("features_path", type=Path, help="Path to the JSON file containing the circuit features.")
    return parser.parse_args()


def align_features(feature_frame, expected_columns):
    """Align prepared features to the set of columns expected by the model."""
    aligned = feature_frame.copy()
    for column in expected_columns:
        if column not in aligned.columns:
            aligned[column] = 0.0
    aligned = aligned[expected_columns]
    return aligned


def predict_runtime(model_path: Path, payload: Dict[str, object]) -> float:
    model = joblib.load(model_path)
    if hasattr(model, "feature_names_in_"):
        expected_columns = list(model.feature_names_in_)
    else:
        expected_columns = model.get_booster().feature_names
    feature_frame = prepare_record(payload)
    # Drop any columns that are not expected by the trained model.
    feature_frame = feature_frame.drop(columns=[col for col in feature_frame.columns if col not in expected_columns], errors="ignore")
    feature_frame = align_features(feature_frame, expected_columns)
    prediction = model.predict(feature_frame)[0]
    return float(prediction)


def main() -> None:
    args = parse_args()
    model_path = args.model_path.resolve()
    with args.features_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    runtime = predict_runtime(model_path, payload)
    print(f"{runtime:.6f}")


if __name__ == "__main__":
    main()
