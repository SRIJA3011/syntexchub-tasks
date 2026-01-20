
# ===============================
# TASK 1 : Salary Prediction
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ===============================
# 1. LOAD DATASET
# ===============================

df = pd.read_csv("Salary_Data.csv")   # <-- ensure filename is correct

print(df.info())
print(df.head())

# ===============================
# 2. HANDLE MISSING VALUES
# ===============================

df.dropna(inplace=True)

# ===============================
# 3. SINGLE FEATURE MODEL
#    Years of Experience → Salary
# ===============================

X_single = df[["Years of Experience"]]
y = df["Salary"]

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_single, y, test_size=0.2, random_state=42
)

single_model = LinearRegression()
single_model.fit(X_train_s, y_train_s)

# 🔴 THIS WAS MISSING IN YOUR CODE
y_pred_single = single_model.predict(X_test_s)

rmse_single = np.sqrt(mean_squared_error(y_test_s, y_pred_single))
r2_single = r2_score(y_test_s, y_pred_single)

print("\nSingle Feature Model")
print("RMSE:", rmse_single)
print("R2 Score:", r2_single)

# ===============================
# 4. MULTIPLE FEATURE MODEL
# ===============================

X_multi = df.drop("Salary", axis=1)
y_multi = df["Salary"]

categorical_features = ["Gender", "Education Level", "Job Title"]
numerical_features = ["Age", "Years of Experience"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numerical_features)
    ]
)

multi_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42
)

multi_model.fit(X_train_m, y_train_m)

y_pred_multi = multi_model.predict(X_test_m)

rmse_multi = np.sqrt(mean_squared_error(y_test_m, y_pred_multi))
r2_multi = r2_score(y_test_m, y_pred_multi)

print("\nMultiple Feature Model")
print("RMSE:", rmse_multi)
print("R2 Score:", r2_multi)

# ===============================
# 5. COMPARISON BAR CHART
# ===============================

models = ["Single Feature", "Multiple Feature"]
rmse_values = [rmse_single, rmse_multi]
r2_values = [r2_single, r2_multi]

plt.figure()
plt.bar(models, rmse_values)
plt.title("RMSE Comparison")
plt.ylabel("RMSE")
plt.show()

plt.figure()
plt.bar(models, r2_values)
plt.title("R2 Score Comparison")
plt.ylabel("R2 Score")
plt.show()

# ===============================
# 6. ACTUAL vs PREDICTED (MULTI)
# ===============================

plt.figure()
plt.scatter(y_test_m, y_pred_multi)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Salary (Multiple Features)")
plt.show()
