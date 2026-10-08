# Titanic Decision Tree Model

This project builds a simple decision tree classifier to predict whether a passenger survived the Titanic disaster based on a few selected features.

## Project Files

- `titanic_decision_tree.py` — loads the Titanic dataset, trains the model, evaluates it, and saves a decision tree visualization.
- `decision_tree.png` — generated model visualization.
- `train.csv` — local dataset file available in the project folder.

## Model Features

The script uses these input fields:

- `Pclass`
- `Sex`
- `Age`
- `Fare`

The target variable is:

- `Survived`

## Requirements

Install the required Python packages:

```bash
pip install pandas matplotlib scikit-learn
```

## Run the Script

```bash
python titanic_decision_tree.py
```

## What the Script Does

1. Loads the Titanic dataset from a public dataset mirror.
2. Encodes the `Sex` column numerically.
3. Fills missing `Age` values with the mean age.
4. Splits the data into training and test sets.
5. Trains a `DecisionTreeClassifier`.
6. Prints model accuracy, confusion matrix, and classification report.
7. Saves the trained decision tree as `decision_tree.png`.

## Notes

- The dataset is loaded from a public GitHub mirror because direct Kaggle access requires authentication.
- If you want to use the local `train.csv` file instead, replace the dataset-loading line in the script with a local file path.
