"""Train per-solver time prediction models using XGBoost and Hyperopt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from feature_utils import ALL_DROP_COLUMNS, prepare_features

SOLVER_TARGETS = ["pono", "rIC3", "ic3ref", "abc"]
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train solver runtime predictors.")
    parser.add_argument("--data-path", type=Path, default=Path("train.csv"), help="Path to the training CSV file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory where models and metrics will be stored.",
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=50,
        help="Maximum number of hyperparameter evaluations for Hyperopt per solver.",
    )
    return parser.parse_args()


def build_feature_matrix(data_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    raw_df = pd.read_csv(data_path)
    features = prepare_features(raw_df)
    if ALL_DROP_COLUMNS:
        print("Excluded invariant or non-informative columns:", ", ".join(ALL_DROP_COLUMNS))
    targets = raw_df[SOLVER_TARGETS].copy()
    return features, targets, list(features.columns)


def create_search_space() -> Dict[str, object]:
    return {
        "max_depth": hp.quniform("max_depth", 3, 8, 1),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)),
        "subsample": hp.uniform("subsample", 0.6, 1.0),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0),
        "reg_lambda": hp.loguniform("reg_lambda", np.log(1e-3), np.log(10.0)),
        "reg_alpha": hp.loguniform("reg_alpha", np.log(1e-5), np.log(1.0)),
        "min_child_weight": hp.uniform("min_child_weight", 1.0, 10.0),
        "gamma": hp.uniform("gamma", 0.0, 5.0),
        "n_estimators": hp.quniform("n_estimators", 200, 1200, 50),
    }


def extract_feature_importance(model: XGBRegressor) -> List[Dict[str, float]]:
    booster = model.get_booster()
    gain_scores = booster.get_score(importance_type="gain")
    if not gain_scores:
        return []
    total_gain = float(sum(gain_scores.values())) or 1.0
    sorted_items = sorted(gain_scores.items(), key=lambda item: item[1], reverse=True)
    return [
        {"feature": feature, "gain": float(score), "gain_fraction": float(score / total_gain)}
        for feature, score in sorted_items
    ]


def evaluate_params(params: Dict[str, float], X: pd.DataFrame, y: pd.Series) -> Tuple[float, float]:
    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=RANDOM_STATE,
        nthread=0,
        **params,
    )

    kfold = KFold(n_splits=min(5, len(X)), shuffle=True, random_state=RANDOM_STATE)
    rmses: List[float] = []
    r2_scores: List[float] = []
    for train_index, valid_index in kfold.split(X):
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        rmse = float(np.sqrt(np.mean((preds - y_valid) ** 2)))
        rmses.append(rmse)
        r2_scores.append(float(r2_score(y_valid, preds)))
    return float(np.mean(rmses)), float(np.mean(r2_scores))


def tune_hyperparameters(X: pd.DataFrame, y: pd.Series, max_evals: int) -> Tuple[Dict[str, float], float, float, Trials]:
    space = create_search_space()

    def objective(trial_params: Dict[str, float]):
        params = {
            "max_depth": int(trial_params["max_depth"]),
            "learning_rate": float(trial_params["learning_rate"]),
            "subsample": float(trial_params["subsample"]),
            "colsample_bytree": float(trial_params["colsample_bytree"]),
            "reg_lambda": float(trial_params["reg_lambda"]),
            "reg_alpha": float(trial_params["reg_alpha"]),
            "min_child_weight": float(trial_params["min_child_weight"]),
            "gamma": float(trial_params["gamma"]),
            "n_estimators": int(trial_params["n_estimators"]),
        }
        rmse, _ = evaluate_params(params, X, y)
        return {"loss": rmse, "status": STATUS_OK}

    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials, rstate=np.random.default_rng(RANDOM_STATE))

    best_params = {
        "max_depth": int(best["max_depth"]),
        "learning_rate": float(best["learning_rate"]),
        "subsample": float(best["subsample"]),
        "colsample_bytree": float(best["colsample_bytree"]),
        "reg_lambda": float(best["reg_lambda"]),
        "reg_alpha": float(best["reg_alpha"]),
        "min_child_weight": float(best["min_child_weight"]),
        "gamma": float(best["gamma"]),
        "n_estimators": int(best["n_estimators"]),
    }
    best_rmse, best_r2 = evaluate_params(best_params, X, y)
    return best_params, best_rmse, best_r2, trials


def train_solver_model(X: pd.DataFrame, y: pd.Series, params: Dict[str, float]) -> XGBRegressor:
    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=RANDOM_STATE,
        nthread=0,
        **params,
    )
    model.fit(X, y)
    return model


def main() -> None:
    args = parse_args()
    X, target_df, feature_columns = build_feature_matrix(args.data_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = args.output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    metrics_report: Dict[str, object] = {}

    for solver in SOLVER_TARGETS:
        y = target_df[solver]
        best_params, best_rmse, best_r2, trials = tune_hyperparameters(X, y, args.max_evals)
        model = train_solver_model(X, y, best_params)

        model.save_model(models_dir / f"{solver}_model.json")

        feature_importance = extract_feature_importance(model)
        metrics_report[solver] = {
            "cv_rmse": best_rmse,
            "cv_r2": best_r2,
            "best_params": best_params,
            "trial_count": len(trials.trials),
            "feature_importance_gain": feature_importance,
        }
        print(f"Trained model for {solver} with CV RMSE={best_rmse:.4f}, CV R2={best_r2:.4f}")
        if feature_importance:
            top_preview = ", ".join(
                f"{item['feature']} ({item['gain_fraction']:.2%})" for item in feature_importance[:5]
            )
            print(f"Top gain features for {solver}: {top_preview}")
        else:
            print(f"Top gain features for {solver}: <no features>")

    metrics_report["feature_columns"] = feature_columns
    metrics_report["dropped_columns"] = ALL_DROP_COLUMNS

    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_report, handle, indent=2)

    print(f"Artifacts saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
