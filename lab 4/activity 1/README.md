# Lab 04 — Activity 1: Diabetes Prediction using Decision Tree Classification

## Objective

To implement a Decision Tree Classification model for predicting whether a patient has diabetes based on medical features.

## Dataset

**Pima Indian Diabetes Dataset** — 768 records with 9 columns.

| Column     | Description                              |
|------------|------------------------------------------|
| pregnant   | Number of times pregnant                 |
| glucose    | Plasma glucose concentration             |
| bp         | Diastolic blood pressure (mm Hg)         |
| skin       | Triceps skin fold thickness (mm)         |
| insulin    | 2-Hour serum insulin (mu U/ml)           |
| bmi        | Body mass index (kg/m²)                  |
| pedigree   | Diabetes pedigree function               |
| age        | Age (years)                              |
| label      | 0 = No Diabetes, 1 = Diabetes            |

## Features Used

- Pregnancy
- Insulin
- BMI
- Age
- Glucose
- Blood Pressure
- Pedigree

## Target

`label` — 0 (No Diabetes) or 1 (Diabetes)

## Algorithm

**Decision Tree Classifier** (scikit-learn)

## Models

| Model              | Parameters                                    | Accuracy |
|--------------------|-----------------------------------------------|----------|
| Original Tree      | `DecisionTreeClassifier()`                    | ~67.53%  |
| Optimized Tree     | `DecisionTreeClassifier(criterion="entropy", max_depth=3)` | ~77.06%  |

## How to Run

### 1. Install required libraries

```bash
pip install pandas scikit-learn matplotlib
```

### 2. Make sure `diabetes.csv` is in the same folder as `activity1.py`

### 3. Run the script

```bash
python activity1.py
```

### 4. Output

- Console prints dataset preview, training/testing split, and accuracy for both models.
- **3 graph windows** will open one by one:
  1. Original Decision Tree visualization
  2. Pruned Decision Tree visualization (max_depth=3, entropy)
  3. Accuracy comparison bar chart

Close each graph window to see the next one.

## Folder Structure

```
activity 1/
├── activity1.py        # Main Python script
├── diabetes.csv        # Pima Indian Diabetes dataset
└── README.md           # This file
```

## Key Concepts

- **Gini Index** — Default splitting criterion for Decision Trees.
- **Entropy / Information Gain** — Alternative splitting criterion; used in the optimized model.
- **Pruning (max_depth)** — Limits tree depth to prevent overfitting and improve generalization.
- **train_test_split** — 70% training, 30% testing with `random_state=1`.

## Tools & Libraries

- Python 3
- pandas
- scikit-learn
- matplotlib
