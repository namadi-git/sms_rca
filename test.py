## DATA PREPROCESSING
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import shap

# Load the dataset
df = pd.read_csv('health_insurance_sms_data.csv')

# Check for missing values
print(df.isnull().sum())

# Impute missing values if any (using median imputation for numeric data)
imputer = SimpleImputer(strategy='median')
df[['active_months_1yr', 'active_months_2yr', 'active_months_25yr']] = imputer.fit_transform(df[['active_months_1yr', 'active_months_2yr', 'active_months_25yr']])

# Feature engineering: Creating new features
df['message_frequency'] = df['cnt_sms_6_month'] / df['active_months_1yr']  # Message frequency
df['message_recency'] = df['days_btwn_first_last_sms']  # Recency of messages

# Convert categorical variables to dummy variables
df = pd.get_dummies(df, columns=['gender', 'marital_status', 'race', 'preferred_language'], drop_first=True)


## 2. Descriptive Analysis
# Calculate opt-out rate by different categories
optout_rate_by_age = df.groupby('age')['opted_out'].mean()
optout_rate_by_race = df.groupby('race')['opted_out'].mean()

# Visualize opt-out rate by categories
plt.figure(figsize=(10,6))
sns.barplot(x=optout_rate_by_age.index, y=optout_rate_by_age.values)
plt.title('Opt-Out Rate by Age')
plt.xlabel('Age')
plt.ylabel('Opt-Out Rate')
plt.show()

plt.figure(figsize=(10,6))
sns.barplot(x=optout_rate_by_race.index, y=optout_rate_by_race.values)
plt.title('Opt-Out Rate by Race')
plt.xlabel('Race')
plt.ylabel('Opt-Out Rate')
plt.show()


## 3. Exploratory Data Analysis (EDA)
# Correlation Matrix
plt.figure(figsize=(15, 10))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Distribution of age and sdoh score
plt.figure(figsize=(10, 5))
sns.histplot(df['age'], kde=True)
plt.title('Distribution of Age')
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(df['sdoh_score'], kde=True)
plt.title('Distribution of SDOH Score')
plt.show()

# Time-based analysis of SMS
plt.figure(figsize=(10, 5))
sns.histplot(df['days_btwn_last_two_sms'], kde=True)
plt.title('Distribution of Days Between Last Two SMS')
plt.show()


## 4. Feature Importance via Predictive Modeling
# Split data into features and target
X = df.drop(['opted_out', 'opted_dt', 'indiv_id', 'last_smsdelivered'], axis=1)
y = df['opted_out']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression Model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Evaluate model
y_pred = log_reg.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix')
plt.show()

# Random Forest Model for feature importance
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Plot Feature Importance
feat_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.title('Top 10 Important Features')
plt.show()

# SHAP values for interpretability
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values[1], X_test)


## 5. Segmentation Analysis
# Segment analysis based on psychographics and sdoh score
df['sdoh_segment'] = pd.qcut(df['sdoh_score'], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
optout_rate_by_sdoh = df.groupby('sdoh_segment')['opted_out'].mean()

# Visualize opt-out rate by SDOH segment
plt.figure(figsize=(10,6))
sns.barplot(x=optout_rate_by_sdoh.index, y=optout_rate_by_sdoh.values)
plt.title('Opt-Out Rate by SDOH Segment')
plt.xlabel('SDOH Segment')
plt.ylabel('Opt-Out Rate')
plt.show()


##6. Behavioral Insights
# Analyze message volume and timing impact on opt-out
plt.figure(figsize=(10,6))
sns.boxplot(x='opted_out', y='cnt_sms_6_month', data=df)
plt.title('Message Volume vs Opt-Out Behavior')
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(x='opted_out', y='avg_length_sms_1_month', data=df)
plt.title('Message Length vs Opt-Out Behavior')
plt.show()

# Analyze engagement patterns
plt.figure(figsize=(10,6))
sns.histplot(df[df['opted_out']==1]['cnt_sms_6_month'], kde=True)
plt.title('Message Frequency for Members Who Opted Out')
plt.show()


##7. Cluster Analysis (Optional)
from sklearn.cluster import KMeans

# Perform K-Means clustering based on demographic and engagement features
kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(df[['age', 'sdoh_score', 'pulse_fsi_score', 'cnt_sms_6_month']])

# Visualize clusters
sns.scatterplot(x='age', y='sdoh_score', hue='cluster', data=df, palette='Set1')
plt.title('K-Means Clustering of Members')
plt.show()


##8. A/B Testing (Implementation Placeholder)
8. A/B Testing (Implementation Placeholder)

##9. Actionable Insights
