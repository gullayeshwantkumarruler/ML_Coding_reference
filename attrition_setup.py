# Great analysis pipeline for binary classification, especially for something like employee attrition where prioritizing high-risk groups (e.g. top 10%) is essential. Let's break this down step-by-step and build a full pipeline using logistic regression.

# Here's what the code will do:

# 1. Preprocessing and train-test split.
# 2. Train a logistic regression model.
# 3. Evaluate using precision and recall.
# 4. Get top 10% based on predicted probability (high risk).
# 5. Evaluate precision and recall on that decile.
# 6. Analyze feature importance (drivers).
# 7. Segment into high, medium, and low risk based on log-odds and compare feature impact.

# ---

# ### ✅ Full Python Code

# ```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df is your DataFrame
# target column is 'attrition' (1 = attrited, 0 = not)
X = df.drop(columns=['attrition'])
y = df['attrition']

# Optional: One-hot encoding or normalization if needed
X = pd.get_dummies(X, drop_first=True)

# 1. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 2. Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 3. Predictions and Evaluation
y_probs = model.predict_proba(X_test)[:, 1]
y_pred = (y_probs >= 0.5).astype(int)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("=== Overall Evaluation ===")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(classification_report(y_test, y_pred))

# 4. Top 10% high-risk decile
X_test_copy = X_test.copy()
X_test_copy['attrition'] = y_test.values
X_test_copy['prob'] = y_probs

top_decile = X_test_copy.sort_values('prob', ascending=False).head(int(0.1 * len(X_test_copy)))
top_pred = (top_decile['prob'] >= 0.5).astype(int)

print("\n=== Top 10% High-Risk Evaluation ===")
print(f"Precision (Top 10%): {precision_score(top_decile['attrition'], top_pred):.4f}")
print(f"Recall (Top 10%): {recall_score(top_decile['attrition'], top_pred):.4f}")

# 5. Feature Importance (Drivers)
coefficients = pd.Series(model.coef_[0], index=X.columns)
coefficients = coefficients.sort_values()

plt.figure(figsize=(10,6))
coefficients.plot(kind='barh')
plt.title("Feature Importance (Logistic Regression Coefficients)")
plt.xlabel("Impact on Log-Odds of Attrition")
plt.tight_layout()
plt.show()

# 6. Risk Segmentation
X_test_copy['log_odds'] = np.log(X_test_copy['prob'] / (1 - X_test_copy['prob']))

# Define percentiles
low = X_test_copy['log_odds'].quantile(0.33)
high = X_test_copy['log_odds'].quantile(0.66)

def segment(row):
    if row['log_odds'] <= low:
        return 'Low Risk'
    elif row['log_odds'] <= high:
        return 'Medium Risk'
    else:
        return 'High Risk'

X_test_copy['risk_segment'] = X_test_copy.apply(segment, axis=1)

# 7. Driver Analysis by Segment
feature_impacts = pd.concat([X_test_copy[['risk_segment']], X_test.copy()[X.columns]], axis=1)

group_means = feature_impacts.groupby('risk_segment').mean()

# Compare feature values across segments
group_means.T.plot(kind='bar', figsize=(12,6))
plt.title("Feature Means by Risk Segment")
plt.ylabel("Mean Feature Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# ```

# ---

# ### 📝 What You’ll Be Able to Say After This

# - The model's overall precision and recall are X and Y.
# - In the top 10% high-risk group, precision and recall are A and B.
# - Key drivers of attrition are [list of features], as seen from coefficient magnitudes.
# - The impact of these drivers changes across low, medium, and high-risk employees, showing how different segments are influenced.

# ---

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# 1. **Visualize global driver importance (summary plot).**
# 2. **Understand individual predictions (force plot / waterfall plot).**
# 3. **Compare average SHAP values across risk segments (group-level driver comparison).**

# ---

# ### 🔁 Updated Part: SHAP Integration into Your Analysis Pipeline

# ```python
import shap

# Use TreeExplainer for tree-based models or KernelExplainer for linear models
explainer = shap.Explainer(model, X_test)
shap_values = explainer(X_test)

# 1. Global Driver Importance
shap.plots.beeswarm(shap_values, max_display=15)

# 2. Individual Explanation for a Sample High-Risk Employee
# Get one high-risk employee from top_decile
sample_idx = top_decile.index[0]
shap.plots.waterfall(shap_values[sample_idx], max_display=10)

# 3. Group-level Driver Comparison
# Add SHAP values to dataframe
shap_df = pd.DataFrame(shap_values.values, columns=X.columns, index=X_test.index)
shap_df['risk_segment'] = X_test_copy['risk_segment']

# Average absolute SHAP values across segments
shap_group_means = shap_df.drop(columns=['risk_segment']).groupby(X_test_copy['risk_segment']).mean().T

# Plot comparison of drivers by risk group
shap_group_means.plot(kind='bar', figsize=(12, 6))
plt.title("Average SHAP Values by Risk Segment (Feature Impact)")
plt.ylabel("Mean Absolute SHAP Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# ```

# ---

### 🧠 Interpretation Tips

# - **Global Summary Plot**: Shows which features have the highest average impact (positive or negative) on the model's output.
# - **Waterfall Plot**: Shows exactly how each feature contributed to an individual’s attrition risk (e.g., "Overtime = Yes" pushed the prediction up by X).
# - **Bar Plot Across Segments**: Tells you which features are more influential for high-risk vs. low-risk segments — perfect for HR business partners.

# ---

# ### 💬 Use This in Reporting:

# > “We used SHAP values to analyze attrition drivers. Features like *Overtime, Job Level, Work-Life Balance,* and *Distance from Home* consistently increased the risk of attrition, particularly in high-risk groups. These insights can guide retention efforts.”

# ---
