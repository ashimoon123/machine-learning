# Salary Prediction Project

This project builds a simple decision tree model to predict whether a salary is more than $100K based on a person's company, job role, and degree.

## Files

- `activity2_salary_prediction.py` — loads the dataset, trains the decision tree, evaluates accuracy, and generates predictions and charts.
- `salaries.csv` — dataset used for training and prediction.
- `decision_tree_salary.png` — saved decision tree visualization produced by the script.
- `salary_prediction_comparison.png` — saved comparison chart for sample predictions.

## Requirements

Make sure you have Python installed and install the required libraries:

```bash
pip install pandas matplotlib scikit-learn
```

## Run the Script

From the project folder, run:

```bash
python activity2_salary_prediction.py
```

## What the Script Does

1. Loads the salary dataset.
2. Encodes categorical text values such as company, job, and degree.
3. Trains a `DecisionTreeClassifier`.
4. Evaluates the model accuracy.
5. Predicts salary outcomes for sample employee profiles.
6. Saves and displays the decision tree and comparison charts.

## Dataset Columns

- `company`
- `job`
- `degree`
- `salary_more_then_100k`

## Example Output

The script prints model accuracy and prediction results for examples such as:

- Google + Computer Engineer + Bachelors
- Google + Computer Engineer + Masters

It also saves visual plots to the project folder.
