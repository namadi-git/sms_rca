"""
To analyze interaction effects of different features on SMS opt-outs, we need to examine how the relationship between multiple variables affects the likelihood of a member opting out. This can be achieved by:

Visualizing the interaction between two features and how they jointly influence the opted_out target.
Using statistical models (e.g., logistic regression with interaction terms) to quantify the interaction effects.
"""
##2. Code for Visualizing Interaction Effects

# Import necessary libraries for interaction effects
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
import itertools

# Interaction between Age and SDOH Score
plt.figure(figsize=(10,6))
sns.scatterplot(x='age', y='sdoh_score', hue='opted_out', data=df, palette='coolwarm')
plt.title('Interaction of Age and SDOH Score on Opt-Out')
plt.show()

# Interaction between cnt_sms_6_month and avg_length_sms_1_month
plt.figure(figsize=(10,6))
sns.scatterplot(x='cnt_sms_6_month', y='avg_length_sms_1_month', hue='opted_out', data=df, palette='coolwarm')
plt.title('Interaction of SMS Frequency and Message Length on Opt-Out')
plt.show()

# 3D plot for interaction between age, cnt_sms_6_month, and opted_out
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['age'], df['cnt_sms_6_month'], df['opted_out'], c=df['opted_out'], cmap='coolwarm')
ax.set_xlabel('Age')
ax.set_ylabel('SMS Count in 6 Months')
ax.set_zlabel('Opt-Out')
plt.title('Interaction of Age, SMS Frequency, and Opt-Out')
plt.show()

# Interaction between cnt_sms_1_week and age
plt.figure(figsize=(10,6))
sns.scatterplot(x='age', y='cnt_sms_1_week', hue='opted_out', data=df, palette='coolwarm')
plt.title('Interaction of Age and Weekly SMS Count on Opt-Out')
plt.show()

# Heatmap for cnt_sms_6_month and sdoh_score interaction
interaction_heatmap = pd.pivot_table(df, values='opted_out', index='cnt_sms_6_month', columns='sdoh_score', aggfunc=np.mean)
plt.figure(figsize=(10,8))
sns.heatmap(interaction_heatmap, cmap='coolwarm', annot=True)
plt.title('Interaction Between SMS Count (6 Months) and SDOH Score on Opt-Out Rate')
plt.show()



### 5. Advanced Visualization: Partial Dependence Plots (PDP)
"""
To better understand interaction effects, Partial Dependence Plots (PDPs) can show the marginal effect of two features on the opt-out likelihood.
"""
from sklearn.inspection import plot_partial_dependence

# Random Forest model to create Partial Dependence Plots
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Plot PDP for age and cnt_sms_6_month interaction
fig, ax = plt.subplots(figsize=(10, 6))
plot_partial_dependence(rf_model, X_train, [('age', 'cnt_sms_6_month')], ax=ax)
plt.title('Partial Dependence Plot: Age and SMS Frequency Interaction')
plt.show()
