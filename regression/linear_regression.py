# ==========================================
# Activity 1: Simple Linear Regression
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ==========================================
# 1. Importing the Dataset
# ==========================================

dataset = pd.read_csv("Salary_Data.csv")

print("==========================================")
print("Original Dataset")
print("==========================================")

print(dataset.head())

print("\nColumns:")
print(dataset.columns)


# ==========================================
# 2. Selecting Required Columns
# ==========================================

# X = Years of Experience
# y = Salary

dataset = dataset[["Years of Experience", "Salary"]]


# ==========================================
# 3. Convert Values to Numeric
# ==========================================

dataset["Years of Experience"] = pd.to_numeric(
    dataset["Years of Experience"],
    errors="coerce"
)

dataset["Salary"] = pd.to_numeric(
    dataset["Salary"],
    errors="coerce"
)


# ==========================================
# 4. Remove Missing Values
# ==========================================

dataset = dataset.dropna()


print("\n==========================================")
print("Cleaned Dataset")
print("==========================================")

print(dataset.head())

print("\nMissing Values:")
print(dataset.isnull().sum())


# ==========================================
# 5. Creating X and y
# ==========================================

X = dataset[["Years of Experience"]].values
y = dataset["Salary"].values


print("\n==========================================")
print("X and y")
print("==========================================")

print("\nX:")
print(X[:5])

print("\ny:")
print(y[:5])


# ==========================================
# 6. Splitting Dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=1 / 3,
    random_state=0
)


print("\n==========================================")
print("Training and Testing Data")
print("==========================================")

print("\nX_train:")
print(X_train)

print("\ny_train:")
print(y_train)

print("\nX_test:")
print(X_test)

print("\ny_test:")
print(y_test)


# ==========================================
# 7. Training Linear Regression Model
# ==========================================

regressor = LinearRegression()

regressor.fit(X_train, y_train)


print("\n==========================================")
print("Model Training")
print("==========================================")

print("Model trained successfully!")


# ==========================================
# 8. Predicting Test Set Results
# ==========================================

y_pred = regressor.predict(X_test)


print("\n==========================================")
print("Actual vs Predicted Salaries")
print("==========================================")

print("\nActual Salaries:")
print(y_test)

print("\nPredicted Salaries:")
print(y_pred)


# ==========================================
# 9. Model Evaluation
# ==========================================

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print("\n==========================================")
print("Model Evaluation")
print("==========================================")

print("Mean Squared Error:", mse)
print("R2 Score:", r2)


# ==========================================
# 10. Visualizing Training Set Results
# ==========================================

plt.figure(figsize=(8, 5))

plt.scatter(
    X_train,
    y_train,
    color="red",
    label="Actual Training Data"
)

# Sort X values for a smooth regression line
X_train_sorted = np.sort(X_train, axis=0)

plt.plot(
    X_train_sorted,
    regressor.predict(X_train_sorted),
    color="blue",
    label="Regression Line"
)

plt.title("Salary vs Experience - Training Set")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.grid(True)

plt.show()


# ==========================================
# 11. Visualizing Test Set Results
# ==========================================

plt.figure(figsize=(8, 5))

plt.scatter(
    X_test,
    y_test,
    color="red",
    label="Actual Test Data"
)

# Sort X values for a smooth regression line
X_test_sorted = np.sort(X_test, axis=0)

plt.plot(
    X_test_sorted,
    regressor.predict(X_test_sorted),
    color="blue",
    label="Regression Line"
)

plt.title("Salary vs Experience - Test Set")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.grid(True)

plt.show()


# ==========================================
# 12. Prediction for 12 Years Experience
# ==========================================

single_prediction = regressor.predict([[12]])


print("\n==========================================")
print("Prediction for 12 Years of Experience")
print("==========================================")

print(
    "Predicted Salary:",
    single_prediction[0]
)


# ==========================================
# 13. Coefficient and Intercept
# ==========================================

coefficient = regressor.coef_[0]
intercept = regressor.intercept_


print("\n==========================================")
print("Linear Regression Parameters")
print("==========================================")

print("Coefficient / Slope:", coefficient)

print("Intercept:", intercept)


# ==========================================
# 14. Regression Equation
# ==========================================

print("\n==========================================")
print("Regression Equation")
print("==========================================")

print(
    f"Salary = {intercept:.2f} + "
    f"({coefficient:.2f} × Years of Experience)"
)


# ==========================================
# 15. Manual Prediction for 15 Years
# ==========================================

manual_prediction = intercept + coefficient * 15


print("\n==========================================")
print("Manual Prediction for 15 Years")
print("==========================================")

print(
    "Manual Prediction:",
    manual_prediction
)


# ==========================================
# 16. Automatic Prediction for 15 Years
# ==========================================

auto_prediction = regressor.predict([[15]])


print("\n==========================================")
print("Automatic Prediction for 15 Years")
print("==========================================")

print(
    "Automatic Prediction:",
    auto_prediction[0]
)


# ==========================================
# 17. Compare Manual and Automatic
# ==========================================

difference = abs(
    manual_prediction - auto_prediction[0]
)


print("\n==========================================")
print("Comparison")
print("==========================================")

print(
    "Manual Prediction:",
    manual_prediction
)

print(
    "Automatic Prediction:",
    auto_prediction[0]
)

print(
    "Difference:",
    difference
)


# ==========================================
# 18. Final Result
# ==========================================

print("\n==========================================")
print("FINAL RESULT")
print("==========================================")

print(
    f"For 12 years of experience, "
    f"predicted salary = {single_prediction[0]:.2f}"
)

print(
    f"For 15 years of experience, "
    f"predicted salary = {auto_prediction[0]:.2f}"
)

print(
    f"R2 Score = {r2:.4f}"
)

print("\nProgram completed successfully!")