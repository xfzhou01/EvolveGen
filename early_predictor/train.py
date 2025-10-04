#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectFromModel
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('train.csv')

# Preprocess frame_clause_counts column: replace list string with its average value
target_col = 'dynamic_features$pdr$frame_clause_counts'

def list_mean(cell):
    if pd.isna(cell):
        return np.nan
    s = str(cell).strip()
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]
    if not s:
        return np.nan
    # Split by comma or whitespace
    parts = [p for p in s.replace(',', ' ').split() if p]
    nums = []
    for p in parts:
        try:
            nums.append(float(p))
        except ValueError:
            pass
    if not nums:
        return np.nan
    return float(np.mean(nums))

if target_col in df.columns:
    df[target_col] = df[target_col].apply(list_mean)

# Identify label columns (last 4 columns are solver times)
label_columns = df.columns[-4:].tolist()
print(f"Label columns (solvers): {label_columns}")

# Define columns to exclude from features
# Exclude identifiers, categorical results, and non-numeric features
exclude_columns = [
    'case_name', 
    'circuit_name',
    'dynamic_features$circuit_name',  # Circuit path (string)
    'dynamic_features$pdr$result',     # Categorical result (SAT/UNSAT/UNDEC)
    'dynamic_features$pdr$frame_clause_counts',  # List/array data (not numeric)
] + label_columns

# Get all feature columns
all_columns = df.columns.tolist()
feature_columns = [col for col in all_columns if col not in exclude_columns]

# print(f"\nTotal columns: {len(all_columns)}")
# print(f"Excluded columns: {len(exclude_columns)}")
# print(f"Feature columns: {len(feature_columns)}")
# print(f"\nFeature columns: {feature_columns}")

# Extract features and labels
X = df[feature_columns].copy()
y_dict = {solver: df[solver].values for solver in label_columns}

# Convert all features to numeric, coerce errors to NaN
print("\nConverting features to numeric...")
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')

# Handle missing values and infinite values in features
print("Handling missing and infinite values...")
X = X.replace([np.inf, -np.inf], np.nan)

# Check for columns with all NaN values
all_nan_cols = X.columns[X.isna().all()].tolist()
if all_nan_cols:
    print(f"Removing columns with all NaN values: {all_nan_cols}")
    X = X.drop(columns=all_nan_cols)

# Fill remaining NaN with median
X = X.fillna(X.median())

# Remove constant features (features with zero variance)
constant_features = X.columns[X.nunique() <= 1].tolist()
if constant_features:
    print(f"Removing constant features: {constant_features}")
    X = X.drop(columns=constant_features)

# Remove features with too many missing values (>50%)
missing_threshold = 0.5
high_missing_features = X.columns[X.isnull().mean() > missing_threshold].tolist()
if high_missing_features:
    print(f"Removing features with >50% missing values: {high_missing_features}")
    X = X.drop(columns=high_missing_features)

print(f"\nFinal feature count: {len(X.columns)}")

# Define hyperparameter search space
space = {
    'eta': hp.uniform('eta', 0.01, 0.3),
    'max_depth': hp.choice('max_depth', range(3, 10)),
    'subsample': hp.uniform('subsample', 0.6, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
    'gamma': hp.uniform('gamma', 0, 0.5),
    'lambda': hp.uniform('lambda', 0, 1),
    'alpha': hp.uniform('alpha', 0, 1)
}

# Number of boosting rounds
num_round = 100

# Dictionary to store trained models
trained_models = {}

# Train a separate model for each solver
for solver_name in label_columns[1:]:
    print(f"\n{'='*60}")
    print(f"Training model for solver: {solver_name}")
    print(f"{'='*60}")
    
    # Get labels for current solver
    y = y_dict[solver_name]
    
    # Handle missing/invalid values in labels
    valid_indices = ~(np.isnan(y) | np.isinf(y))
    X_valid = X[valid_indices].copy()
    y_valid = y[valid_indices]
    
    if len(y_valid) < 10:
        print(f"Warning: Not enough valid samples for {solver_name}, skipping...")
        continue
    
    # Split train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_valid, y_valid, test_size=0.1, random_state=42
    )
    
    # Initial feature selection using L1 regularization
    print("Performing initial feature selection...")
    selector = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_estimators=50,
        reg_alpha=1.0,
        reg_lambda=1.0
    )
    selector.fit(X_train, y_train)
    
    # Select features based on importance threshold
    selection_model = SelectFromModel(selector, prefit=True, threshold='median')
    X_train_selected = selection_model.transform(X_train)
    X_test_selected = selection_model.transform(X_test)
    
    # Get selected feature names
    selected_features = X_train.columns[selection_model.get_support()].tolist()
    print(f"Selected {len(selected_features)} features out of {len(X_train.columns)}")
    
    # Convert to DMatrix format
    dtrain = xgb.DMatrix(X_train_selected, label=y_train)
    dtest = xgb.DMatrix(X_test_selected, label=y_test)
    
    # Define objective function for hyperparameter optimization
    def objective(params):
        params['objective'] = 'reg:squarederror'
        params['eval_metric'] = 'rmse'
        params['nthread'] = 4
        params['seed'] = 42
        
        model = xgb.train(
            params, dtrain, num_round, 
            evals=[(dtest, 'eval')],
            verbose_eval=False
        )
        
        y_pred_test = model.predict(dtest)
        r2_test = r2_score(y_test, y_pred_test)
        
        return {'loss': -r2_test, 'status': STATUS_OK}
    
    # Create Trials object for tracking optimization
    trials = Trials()
    
    # Perform hyperparameter optimization
    print("Optimizing hyperparameters...")
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials,
        verbose=False
    )
    
    print(f"Best hyperparameters: {best}")
    
    # Train final model with best hyperparameters
    best_params = {
        'eta': best['eta'],
        'max_depth': best['max_depth'],
        'subsample': best['subsample'],
        'colsample_bytree': best['colsample_bytree'],
        'gamma': best['gamma'],
        'lambda': best['lambda'],
        'alpha': best['alpha'],
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'nthread': 4,
        'seed': 42
    }
    
    final_model = xgb.train(
        best_params, dtrain, num_round, 
        evals=[(dtest, 'eval')],
        verbose_eval=False
    )
    
    # Evaluate final model
    y_pred_train = final_model.predict(dtrain)
    y_pred_test = final_model.predict(dtest)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    print(f"\nFinal Results for {solver_name}:")
    print(f"  Train R2 score: {r2_train:.4f}")
    print(f"  Test R2 score: {r2_test:.4f}")
    print(f"  Train RMSE: {rmse_train:.4f}")
    print(f"  Test RMSE: {rmse_test:.4f}")
    
    # Store model and selected features
    trained_models[solver_name] = {
        'model': final_model,
        'selected_features': selected_features,
        'best_params': best_params,
        'metrics': {
            'r2_train': r2_train,
            'r2_test': r2_test,
            'rmse_train': rmse_train,
            'rmse_test': rmse_test
        }
    }
    
    # Save model
    model_filename = f'model_{solver_name}.json'
    final_model.save_model(model_filename)
    print(f"Model saved to {model_filename}")

# Print summary
print(f"\n{'='*60}")
print("Training Summary")
print(f"{'='*60}")
for solver_name, model_info in trained_models.items():
    print(f"\n{solver_name}:")
    print(f"  Number of features: {len(model_info['selected_features'])}")
    print(f"  Test R2: {model_info['metrics']['r2_test']:.4f}")
    print(f"  Test RMSE: {model_info['metrics']['rmse_test']:.4f}")

# Save feature information
import json
feature_info = {
    solver: {
        'selected_features': info['selected_features'],
        'best_params': info['best_params'],
        'metrics': info['metrics']
    }
    for solver, info in trained_models.items()
}

with open('model_info.json', 'w') as f:
    json.dump(feature_info, f, indent=2)
print("\nModel information saved to model_info.json")