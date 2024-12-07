import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from datetime import timedelta

# Assuming aggregate_member_data function is already defined

# Step 1: Load and Aggregate Data
def prepare_data(df):
    df['Smsdelivered_dt'] = pd.to_datetime(df['Smsdelivered_dt'])
    aggregated_data = aggregate_member_data(df)
    aggregated_data['Opted_out'] = aggregated_data['Opted_out'].astype(int)  # Ensure binary format
    return aggregated_data

# Step 2: Generate Rolling Window Features
def generate_rolling_features(df, reference_date, window):
    window_data = df[df['Smsdelivered_dt'] <= reference_date].copy()
    window_data = window_data[window_data['Smsdelivered_dt'] > reference_date - timedelta(days=window)]
    aggregated_features = aggregate_member_data(window_data)
    return aggregated_features

# Step 3: Define Rolling Window Cross-Validation
def rolling_window_cross_validation(df, train_window, test_window, step=30):
    df = df.sort_values('Smsdelivered_dt')
    train_test_splits = []

    min_date = df['Smsdelivered_dt'].min()
    max_date = df['Smsdelivered_dt'].max()

    current_start = min_date

    while current_start + timedelta(days=train_window + test_window) <= max_date:
        train_end = current_start + timedelta(days=train_window)
        test_end = train_end + timedelta(days=test_window)

        train = df[(df['Smsdelivered_dt'] >= current_start) & (df['Smsdelivered_dt'] < train_end)]
        test = df[(df['Smsdelivered_dt'] >= train_end) & (df['Smsdelivered_dt'] < test_end)]

        train_test_splits.append((train, test))
        current_start += timedelta(days=step)

    return train_test_splits

# Step 4: Model Training and Evaluation
def train_and_evaluate(df, train_window=180, test_window=30, step=30):
    # Generate train-test splits
    splits = rolling_window_cross_validation(df, train_window, test_window, step)

    metrics = []
    for i, (train, test) in enumerate(splits):
        X_train = train.drop(columns=['Subscriberkey', 'Opted_out', 'Smsdelivered_dt'])
        y_train = train['Opted_out']
        X_test = test.drop(columns=['Subscriberkey', 'Opted_out', 'Smsdelivered_dt'])
        y_test = test['Opted_out']

        # Initialize and fit XGBoost
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Evaluate performance
        auc = roc_auc_score(y_test, y_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        metrics.append({'fold': i+1, 'auc': auc, 'precision': precision, 'recall': recall, 'f1': f1})

    return pd.DataFrame(metrics)

# Step 5: Interpret Results
# Run the process
data = pd.read_csv('your_data.csv')  # Replace with your actual data
prepared_data = prepare_data(data)
results = train_and_evaluate(prepared_data)

print(results)
