import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load the dataset
dataset = pd.read_csv('Position_salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Linear Regression model for comparison
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Training the Polynomial Regression model
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X)[0], max(X)[0], 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='red', label='Actual Salaries')
plt.plot(X, lin_reg.predict(X), color='green', label='Linear Regression Fit')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue', label='Polynomial Regression Fit (Degree=4)')

plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.legend()
plt.savefig('polynomial_regression_plot.png')
plt.close()
