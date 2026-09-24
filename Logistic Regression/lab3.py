import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download dataset
url = "https://raw.githubusercontent.com/ucg8j/kaggle_HR/master/HR_comma_sep.csv"
file_path = "HR_comma_sep.csv"
if not os.path.exists(file_path):
    print("Downloading dataset...")
    urllib.request.urlretrieve(url, file_path)
    print("Downloaded.")
else:
    print("Dataset already exists.")

df = pd.read_csv(file_path)

# 1. Exploratory Data Analysis
print("\n--- 1. Exploratory Data Analysis ---")
print("Missing values:")
print(df.isnull().sum())
print("\nAverage values for employees who left vs stayed:")
# group by numeric columns to find impact
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
print(df.groupby('left')[numeric_cols].mean())

# 2. Plot bar charts showing impact of employee salaries on retention
plt.figure(figsize=(8, 6))
pd.crosstab(df['salary'], df['left']).plot(kind='bar', stacked=False)
plt.title('Employee Retention by Salary')
plt.xlabel('Salary Level')
plt.ylabel('Number of Employees')
plt.legend(['Stayed (0)', 'Left (1)'])
plt.tight_layout()
plt.savefig('salary_retention.png')
print("\nSaved 'salary_retention.png'")

# 3. Plot bar charts showing correlation between department and employee retention
plt.figure(figsize=(10, 6))
# Note: column might be named 'sales' or 'Department' in different versions of the dataset
dept_col = 'Department' if 'Department' in df.columns else 'sales'
pd.crosstab(df[dept_col], df['left']).plot(kind='bar', stacked=False)
plt.title('Employee Retention by Department')
plt.xlabel('Department')
plt.ylabel('Number of Employees')
plt.legend(['Stayed (0)', 'Left (1)'])
plt.tight_layout()
plt.savefig('department_retention.png')
print("Saved 'department_retention.png'")

# 4. Build logistic regression model using variables narrowed down
# Narrowed down to: satisfaction_level, average_montly_hours, promotion_last_5years, and salary
subdf = df[['satisfaction_level', 'average_montly_hours', 'promotion_last_5years', 'salary']]
salary_dummies = pd.get_dummies(subdf['salary'], prefix='salary')
X = pd.concat([subdf.drop('salary', axis=1), salary_dummies], axis=1)
y = df['left']

print("\n--- 4. Building Logistic Regression Model ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 5. Measure accuracy
print("\n--- 5. Measuring Accuracy ---")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
