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
def tune_model_with_rolling_validation(df, train_window=360, test_window=90, step=180):
    # Get rolling splits
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
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10)
        }

        scores = []

        # Evaluate on rolling splits
        for train, test in splits:
            X_train = train.drop(columns=['Subscriberkey', 'Opted_out', 'Smsdelivered_dt'])
            y_train = train['Opted_out']
            X_test = test.drop(columns=['Subscriberkey', 'Opted_out', 'Smsdelivered_dt'])
            y_test = test['Opted_out']

            # Train model
            model = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
            model.fit(X_train, y_train)

            # Predict and evaluate
            preds = model.predict_proba(X_test)[:, 1]
            score = roc_auc_score(y_test, preds)
            scores.append(score)

        # Return the average AUC across folds
        return np.mean(scores)

    # Perform optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    # Return the best hyperparameters
    return study.best_params

# Train the Best Model
def train_best_model(df, best_params, train_window=360, test_window=90, step=180):
    splits = rolling_window_cross_validation(df, train_window, test_window, step)
    all_scores = []

    for train, test in splits:
        X_train = train.drop(columns=['Subscriberkey', 'Opted_out', 'Smsdelivered_dt'])
        y_train = train['Opted_out']
        X_test = test.drop(columns=['Subscriberkey', 'Opted_out', 'Smsdelivered_dt'])
        y_test = test['Opted_out']

        # Train model
        model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)

        preds = model.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, preds)
        all_scores.append(score)

    avg_score = np.mean(all_scores)
    print(f"Average AUC across folds: {avg_score:.4f}")

    # Train the model on the entire dataset
    X_full = df.drop(columns=['Subscriberkey', 'Opted_out', 'Smsdelivered_dt'])
    y_full = df['Opted_out']
    final_model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')
    final_model.fit(X_full, y_full)

    return final_model

# Save the Model
import joblib

def save_model(model, file_name='best_model.pkl'):
    joblib.dump(model, file_name)
    print(f"Model saved to {file_name}")
