
# Define a Function to Tune a Single Model
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

def tune_single_model(X, y, n_trials=50):
    """
    Tune an XGBoost model using Optuna.

    Parameters:
    - X: Features for training.
    - y: Target variable for training.
    - n_trials: Number of hyperparameter optimization trials.

    Returns:
    - best_params: Best hyperparameters found by Optuna.
    """
    def objective(trial):
        # Define hyperparameter search space
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 1),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10),
        }

        # Train XGBoost model
        model = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
        model.fit(X, y)

        # Predict probabilities
        y_pred_proba = model.predict_proba(X)[:, 1]

        # Evaluate using AUC
        auc = roc_auc_score(y, y_pred_proba)
        return auc

    # Optimize hyperparameters
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    return study.best_params

# 2. Integrate Tuning into the Ensemble Training

from datetime import timedelta
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

def train_time_based_ensemble_dynamic(df, target_column='Opted_out', n_trials=30):
    """
    Train a time-based ensemble model with dynamic time windows based on each member's last SMS date.
    
    Parameters:
    - df: DataFrame containing features and target.
    - target_column: Name of the column containing the binary target variable.
    - n_trials: Number of hyperparameter optimization trials for each model.

    Returns:
    - models: List of trained models for 6, 12, and 24 months.
    - weights: List of weights for the ensemble.
    - predictions: Combined predictions from the ensemble.
    - metrics: Dictionary of evaluation metrics for the ensemble.
    - best_params: List of best hyperparameters for each model.
    """
    # Define time windows and weights
    time_windows = [6, 12, 24]  # in months
    weights = [0.5, 0.3, 0.2]

    # Initialize models, predictions, and best hyperparameters
    models = []
    weighted_predictions = []
    best_params = []

    # Calculate the last SMS date for each member
    last_sms_dates = df.groupby('Subscriberkey')['Smsdelivered_dt'].max()

    # Iterate over each Subscriberkey and calculate their personalized time windows
    for window, weight in zip(time_windows, weights):
        # Filter data for the current time window for each member
        filtered_rows = []
        for subscriber, last_sms_date in last_sms_dates.items():
            start_date = last_sms_date - timedelta(days=window * 30)
            member_data = df[(df['Subscriberkey'] == subscriber) & (df['Smsdelivered_dt'] >= start_date)]
            filtered_rows.append(member_data)
        
        filtered_df = pd.concat(filtered_rows)

        # Split features and target
        X = filtered_df.drop(columns=['Subscriberkey', 'Smsdelivered_dt', target_column])
        y = filtered_df[target_column]

        # Tune model
        params = tune_single_model(X, y, n_trials=n_trials)
        best_params.append(params)

        # Train XGBoost model with best parameters
        model = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
        model.fit(X, y)
        models.append(model)

        # Predict probabilities
        y_pred_proba = model.predict_proba(X)[:, 1]

        # Apply weight to predictions
        weighted_predictions.append(y_pred_proba * weight)

    # Combine weighted predictions
    combined_predictions = sum(weighted_predictions)

    # Evaluate the ensemble
    X_full = df.drop(columns=['Subscriberkey', 'Smsdelivered_dt', target_column])
    y_full = df[target_column]

    # Convert probabilities to binary predictions for evaluation
    binary_predictions = (combined_predictions >= 0.5).astype(int)

    metrics = {
        'auc': roc_auc_score(y_full, combined_predictions),
        'precision': precision_score(y_full, binary_predictions),
        'recall': recall_score(y_full, binary_predictions),
        'f1': f1_score(y_full, binary_predictions)
    }

    return models, weights, combined_predictions, metrics, best_params
