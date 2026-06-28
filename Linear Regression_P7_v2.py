

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns


# -------------------------------
# 1. Load Dataset
# -------------------------------
data = fetch_california_housing()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="Price")


# -------------------------------
# 2. Train-Test Split (BEFORE scaling)
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -------------------------------
# 3. Create Pipeline (Prevents Leakage)
# -------------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),     # Fit only on training data internally
    ("model", LinearRegression())
])


# -------------------------------
# 4. Train Model
# -------------------------------
pipeline.fit(X_train, y_train)


# -------------------------------
# 5. Predict
# -------------------------------
y_pred = pipeline.predict(X_test)

print("First 5 Predictions:", y_pred[:5])


# -------------------------------
# 6. Manual Metric Calculations
# -------------------------------

# Residual Sum of Squares (SSE)
SSE = np.sum((y_test - y_pred) ** 2)

# Mean Squared Error
MSE = np.mean((y_test - y_pred) ** 2)

# Total Sum of Squares (TSS)
TSS = np.sum((y_test - np.mean(y_test)) ** 2)

# Manual R²
R2_manual = 1 - (SSE / TSS)


# -------------------------------
# 7. Sklearn Metrics
# -------------------------------
R2_sklearn = r2_score(y_test, y_pred)
MSE_sklearn = mean_squared_error(y_test, y_pred)


# -------------------------------
# 8. Print Results
# -------------------------------
print(f"\nManual R²: {R2_manual:.4f}")
print(f"Sklearn R²: {R2_sklearn:.4f}")
print(f"SSE: {SSE:.4f}")
print(f"MSE (manual): {MSE:.4f}")
print(f"MSE (sklearn): {MSE_sklearn:.4f}")


# Calculate residuals
residuals = y_test - y_pred

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residual Plot: California Housing')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals (Actual - Predicted)')
plt.show()
