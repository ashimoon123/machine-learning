# Car Evaluation Decision Tree Predictor

This project contains a Machine Learning pipeline to predict the safety and overall acceptability of a car based on its features using a Decision Tree Classifier. It utilizes the [Car Evaluation Data Set](http://archive.ics.uci.edu/ml/datasets/Car+Evaluation) from the UCI Machine Learning Repository.

## 📁 Files
- **`car_evaluation_dt.py`**: A command-line Python script that downloads the dataset, trains the Decision Tree model, and evaluates it on a test set (printing accuracy and the classification report).
- **`car_evaluation_gui.py`**: A Graphical User Interface (GUI) application built with `tkinter` that allows users to interactively select car features and get real-time predictions.

## ⚙️ Requirements
The project relies on a few standard Python packages for data manipulation and machine learning:
- `pandas`
- `scikit-learn`

*(Note: `tkinter` is built-in with Python on Windows, so no installation is required for the GUI).*

If you haven't installed the dependencies, you can do so via pip:
```bash
pip install pandas scikit-learn
```

## 🚀 How to Run

### 1. Command-Line Evaluation Script
Run the evaluation script in your terminal to see the model's accuracy:
```bash
python car_evaluation_dt.py
```
*Expected Output: Accuracy ~96.7% and a detailed classification report.*

### 2. GUI Application
Run the interactive Graphical User Interface to test the model yourself:
```bash
python car_evaluation_gui.py
```
This will open a window where you can select the car's attributes (buying price, maintenance cost, number of doors, persons capacity, luggage boot size, and safety level) to predict if the car is Unacceptable, Acceptable, Good, or Very Good.
