import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the data
df = pd.read_csv('hiring.csv')

# Preprocessing
# 1. Fill missing experience with 'zero' and map text to number
word_to_num = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
    'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
    'ten': 10, 'eleven': 11
}

df['experience'] = df['experience'].fillna('zero')
df['experience'] = df['experience'].map(word_to_num)

# 2. Fill missing test_score with the median
import math
median_test_score = math.floor(df['test_score(out of 10)'].median())
df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(median_test_score)

# Define X and y
X = df[['experience', 'test_score(out of 10)', 'interview_score(out of 10)']]
y = df['salary($)']

# Create and train the model
model = LinearRegression()
model.fit(X.values, y)

# Predictions
# Candidate 1: 2 yr experience, 9 test score, 6 interview score
pred_1 = model.predict([[2, 9, 6]])
print(f"Predicted salary for Candidate 1 (2 yrs, 9 test, 6 interview): ${pred_1[0]:.2f}")

# Candidate 2: 12 yr experience, 10 test score, 10 interview score
pred_2 = model.predict([[12, 10, 10]])
print(f"Predicted salary for Candidate 2 (12 yrs, 10 test, 10 interview): ${pred_2[0]:.2f}")
