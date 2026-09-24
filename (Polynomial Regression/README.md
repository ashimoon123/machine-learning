# Polynomial Regression Example

This project demonstrates how Polynomial Regression overcomes the limitations of simple Linear Regression when dealing with non-linear datasets.

## Problem Statement
Simple Linear Regression only works well when the relationship between the independent and dependent variables is linear. When the data has curves or fluctuations, drawing a straight line through it fails to capture the true trend (resulting in underfitting).

## Solution
Polynomial Regression solves this by taking our independent features (like $X$) and generating higher-degree polynomial terms (like $X^2, X^3, X^4$). This essentially maps our linear algorithm into a flexible, curvilinear relationship that molds perfectly to the non-linear data points.

## Files in this Directory

* **`Position_salaries.csv`**: A standard dataset containing Position Levels (1 to 10) and their corresponding expected Salaries. This data forms an exponential curve (non-linear).
* **`poly_reg.py`**: The Python script that trains both a Linear Regression model and a Polynomial Regression model for comparison, and plots the results.
* **`polynomial_regression_plot.png`**: The output graph visualizing how the Polynomial fit compares to the Linear fit.

## How to Run

1. **Install dependencies**: Ensure you have Python installed, then install the required libraries:
   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```

2. **Execute the script**:
   Navigate to this directory in your terminal and run:
   ```bash
   python poly_reg.py
   ```

3. **View the results**: 
   The script will generate or update the `polynomial_regression_plot.png` image file in this folder showing the fitted models.
