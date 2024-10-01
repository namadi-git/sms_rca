#xgboost classificationimport numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Define features and target
X = df.drop(columns=['opted_out'])  # Dropping the target variable
y = df['opted_out']  # Target variable

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Apply SMOTE for oversampling the minority class
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Initialize the XGBoost Classifier
xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

# Define the hyperparameters grid for randomized search
param_dist = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4],
    'scale_pos_weight': [1, 2, 5, 10, 20]  # For class imbalance
}

# Randomized search with 5-fold cross-validation
random_search = RandomizedSearchCV(estimator=xgb,
                                   param_distributions=param_dist,
                                   n_iter=100,  # Number of random configurations to try
                                   scoring='roc_auc',  # Use AUC as the evaluation metric
                                   cv=5,  # 5-fold cross-validation
                                   verbose=1,
                                   random_state=42,
                                   n_jobs=-1)  # Use all available cores

# Fit the model using RandomizedSearchCV on the resampled training data
random_search.fit(X_train_resampled, y_train_resampled)

# Predict on the test set
y_pred = random_search.best_estimator_.predict(X_test)

# Classification report
print(classification_report(y_test, y_pred))

# ROC AUC score
roc_auc = roc_auc_score(y_test, random_search.best_estimator_.predict_proba(X_test)[:, 1])
print(f'ROC AUC Score: {roc_auc}')

print(f'Best Hyperparameters: {random_search.best_params_}')


import matplotlib.pyplot as plt
import xgboost as xgb

# Extract the best XGBoost model from RandomizedSearchCV
best_xgb_model = random_search.best_estimator_

# Plot feature importance using XGBoost's built-in method
xgb.plot_importance(best_xgb_model, importance_type='weight', max_num_features=10)
plt.title('Top 10 Features by Tree-Based Importance')
plt.show()



import shap

# Create an explainer object using the best XGBoost model
explainer = shap.Explainer(best_xgb_model)

# Compute SHAP values for the test set
shap_values = explainer.shap_values(X_test)

# Plot the summary plot showing feature importance using SHAP values
shap.summary_plot(shap_values, X_test, plot_type="bar", show=True)

# Plot SHAP summary (detailed plot)
shap.summary_plot(shap_values, X_test)


