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
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import numpy as np

def train_model_with_rolling_validation(df, best_params, train_window=360, test_window=90, step=180):
    # Perform rolling validation splits
    splits = rolling_window_cross_validation(df, train_window, test_window, step)

    # Initialize lists for metrics
    train_metrics_per_fold = []
    test_metrics_per_fold = []

    for fold, (train, test) in enumerate(splits, start=1):
        # Prepare training and testing datasets
        X_train = train.drop(columns=['Subscriberkey', 'Opted_out', 'Smsdelivered_dt'])
        y_train = train['Opted_out']
        X_test = test.drop(columns=['Subscriberkey', 'Opted_out', 'Smsdelivered_dt'])
        y_test = test['Opted_out']

        # Train the model
        model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)

        # Train predictions and metrics
        y_train_pred = model.predict(X_train)
        y_train_proba = model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, y_train_proba)
        train_precision = precision_score(y_train, y_train_pred)
        train_recall = recall_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred)

        # Test predictions and metrics
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_test_proba)
        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)

        # Append metrics for this fold
        train_metrics_per_fold.append({
            'fold': fold,
            'auc': train_auc,
            'precision': train_precision,
            'recall': train_recall,
            'f1': train_f1
        })
        test_metrics_per_fold.append({
            'fold': fold,
            'auc': test_auc,
            'precision': test_precision,
            'recall': test_recall,
            'f1': test_f1
        })

    # Calculate average metrics across folds
    avg_train_metrics = {key: np.mean([m[key] for m in train_metrics_per_fold]) for key in train_metrics_per_fold[0] if key != 'fold'}
    avg_test_metrics = {key: np.mean([m[key] for m in test_metrics_per_fold]) for key in test_metrics_per_fold[0] if key != 'fold'}

    # Train final model on the entire dataset
    X_full = df.drop(columns=['Subscriberkey', 'Opted_out', 'Smsdelivered_dt'])
    y_full = df['Opted_out']
    final_model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')
    final_model.fit(X_full, y_full)

    # Return final model, fold metrics, and averages
    return {
        'final_model': final_model,
        'train_metrics_per_fold': train_metrics_per_fold,
        'test_metrics_per_fold': test_metrics_per_fold,
        'avg_train_metrics': avg_train_metrics,
        'avg_test_metrics': avg_test_metrics,
        'best_params': best_params
    }
