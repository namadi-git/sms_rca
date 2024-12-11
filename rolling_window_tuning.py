import optuna
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from datetime import timedelta

# Rolling Window Cross-Validation
def rolling_window_cross_validation(df, train_window, test_window, step):
    df = df.sort_values('Smsdelivered_dt')
    splits = []

    min_date = df['Smsdelivered_dt'].min()
    max_date = df['Smsdelivered_dt'].max()

    current_start = min_date

    while current_start + timedelta(days=train_window + test_window) <= max_date:
        train_end = current_start + timedelta(days=train_window)
        test_end = train_end + timedelta(days=test_window)

        train = df[(df['Smsdelivered_dt'] >= current_start) & (df['Smsdelivered_dt'] < train_end)]
        test = df[(df['Smsdelivered_dt'] >= train_end) & (df['Smsdelivered_dt'] < test_end)]

        splits.append((train, test))
        current_start += timedelta(days=step)

    return splits

# Hyperparameter Tuning with Optuna
from sklearn.metrics import roc_auc_score, precision_score, recall_score

def tune_model_with_rolling_validation(df, train_window=360, test_window=90, step=180, n_trials=50):
    """
    Perform hyperparameter tuning with rolling validation using Optuna.

    Parameters:
    - df: DataFrame with input data.
    - train_window: Number of days in the training window.
    - test_window: Number of days in the testing window.
    - step: Step size for rolling splits.
    - n_trials: Number of hyperparameter optimization trials.

    Returns:
    - best_params: The best hyperparameters from the Optuna study.
    - fold_metrics: List of metrics (AUC, Precision, Recall) for training and testing for each fold.
    - avg_metrics: Average metrics across all folds.
    """
    # Get rolling validation splits
    splits = rolling_window_cross_validation(df, train_window, test_window, step)

    def objective(trial):
        # Define hyperparameter search space
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 1),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10),
        }

        # Initialize AUC scores for this trial
        fold_aucs = []

        for train, test in splits:
            # Prepare training and testing datasets
            X_train = train.drop(columns=['Subscriberkey', 'Opted_out', 'Smsdelivered_dt'])
            y_train = train['Opted_out']
            X_test = test.drop(columns=['Subscriberkey', 'Opted_out', 'Smsdelivered_dt'])
            y_test = test['Opted_out']

            # Train the model
            model = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
            model.fit(X_train, y_train)

            # Predict probabilities for evaluation
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Calculate AUC for this split
            auc = roc_auc_score(y_test, y_pred_proba)
            fold_aucs.append(auc)

        # Return the average AUC across all folds
        return np.mean(fold_aucs)

    # Perform optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    # Get the best parameters
    best_params = study.best_params

    # Evaluate the best parameters on the splits
    fold_metrics = []
    for train, test in splits:
        X_train = train.drop(columns=['Subscriberkey', 'Opted_out', 'Smsdelivered_dt'])
        y_train = train['Opted_out']
        X_test = test.drop(columns=['Subscriberkey', 'Opted_out', 'Smsdelivered_dt'])
        y_test = test['Opted_out']

        # Train the model with the best parameters
        model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)

        # Train metrics
        y_train_pred = model.predict(X_train)
        y_train_proba = model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, y_train_proba)
        train_precision = precision_score(y_train, y_train_pred)
        train_recall = recall_score(y_train, y_train_pred)

        # Test metrics
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_test_proba)
        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)

        # Append metrics for this fold
        fold_metrics.append({
            'train_auc': train_auc,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'test_auc': test_auc,
            'test_precision': test_precision,
            'test_recall': test_recall
        })

    # Calculate average metrics across folds
    avg_metrics = {key: np.mean([m[key] for m in fold_metrics]) for key in fold_metrics[0]}

    # Output results
    print("Best Parameters:", best_params)
    print("Average Metrics Across Folds:", avg_metrics)

    return best_params, fold_metrics, avg_metrics
