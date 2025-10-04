"""Utility helpers for feature preparation and transformation."""
from __future__ import annotations

import json
from typing import Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd


FRAME_COUNTS_COL = "dynamic_features$pdr$frame_clause_counts"
SOLVER_RUNTIME_COLUMNS = {"pono", "rIC3", "ic3ref", "abc"}
# Columns that do not carry predictive power in raw form and should be dropped in both train and predict.
DROP_COLUMNS = {
    "case_name",
    "circuit_name",
    "dynamic_features$circuit_name",
    "dynamic_features$pdr$result",
    FRAME_COUNTS_COL,
}
DROP_COLUMNS |= SOLVER_RUNTIME_COLUMNS
# Columns that were constant across the training corpus; removing them keeps the feature space tight.
CONSTANT_COLUMNS = {
    "dynamic_features$pdr$params$fMonoCnf",
    "dynamic_features$pdr$params$fSkipGeneral",
    "dynamic_features$pdr$params$fSolveAll",
    "dynamic_features$pdr$params$nConfLimit",
    "dynamic_features$pdr$params$nFrameMax",
    "dynamic_features$pdr$params$nRecycle",
    "dynamic_features$pdr$params$nRestLimit",
    "dynamic_features$pdr$params$nTimeOut",
    "dynamic_features$pdr$summary$nCex",
    "dynamic_features$pdr$summary$nQueMax",
    "frame_clause_first",
    "frame_clause_min",
}
DROP_COLUMNS |= CONSTANT_COLUMNS


ALL_DROP_COLUMNS = sorted(DROP_COLUMNS)


def _ensure_array(values: Iterable[float]) -> np.ndarray:
    """Convert an iterable of values into a 1D numpy array of floats."""
    arr = np.array(list(values), dtype=float)
    if arr.size == 0:
        return np.array([0.0], dtype=float)
    return arr


def parse_frame_clause_counts(raw_value: object) -> List[float]:
    """Parse frame clause counts stored as a JSON-like string into floats."""
    if isinstance(raw_value, list):
        return [float(v) for v in raw_value]
    if isinstance(raw_value, str):
        value = raw_value.strip()
        if not value:
            return []
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [float(v) for v in parsed]
        except json.JSONDecodeError:
            cleaned = value.strip("[]")
            if not cleaned:
                return []
            return [float(piece.strip()) for piece in cleaned.split(",") if piece.strip()]
    if pd.isna(raw_value):
        return []
    return [float(raw_value)]


def frame_clause_statistics(values: Iterable[float]) -> Dict[str, float]:
    """Generate descriptive statistics for the frame clause counts."""
    arr = _ensure_array(values)
    return {
        "frame_clause_len": float(arr.size),
        "frame_clause_sum": float(arr.sum()),
        "frame_clause_mean": float(arr.mean()),
        "frame_clause_std": float(arr.std(ddof=0)),
        "frame_clause_min": float(arr.min()),
        "frame_clause_max": float(arr.max()),
        "frame_clause_median": float(np.median(arr)),
        "frame_clause_nonzero": float(np.count_nonzero(arr)),
        "frame_clause_first": float(arr[0]),
        "frame_clause_last": float(arr[-1]),
        "frame_clause_range": float(arr.max() - arr.min()),
    }


def flatten_record(record: Mapping[str, object], prefix: str = "") -> Dict[str, object]:
    """Flatten nested dictionaries using `$` as separator."""
    flat: Dict[str, object] = {}
    for key, value in record.items():
        qualified = f"{prefix}${key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(flatten_record(value, qualified))
        else:
            flat[qualified] = value
    return flat


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic feature engineering and drop unused columns."""
    processed = df.copy()
    if FRAME_COUNTS_COL not in processed:
        processed[FRAME_COUNTS_COL] = [[] for _ in range(len(processed))]
    frame_features = processed[FRAME_COUNTS_COL].apply(parse_frame_clause_counts).apply(frame_clause_statistics)
    frame_df = pd.DataFrame(list(frame_features))
    processed = pd.concat([processed.drop(columns=[FRAME_COUNTS_COL]), frame_df], axis=1)

    processed = processed.drop(columns=[col for col in DROP_COLUMNS if col in processed.columns])

    for column in processed.columns:
        if processed[column].dtype == object:
            processed[column] = pd.to_numeric(processed[column], errors="coerce").fillna(0.0)

    processed = processed.fillna(0.0)
    return processed[sorted(processed.columns)]


def prepare_record(record: Mapping[str, object]) -> pd.DataFrame:
    """Prepare a single record using the shared feature pipeline."""
    flat = flatten_record(record)
    frame = pd.DataFrame([flat])
    return prepare_features(frame)
