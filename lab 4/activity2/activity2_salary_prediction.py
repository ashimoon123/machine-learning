import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn import tree

# -----------------------------------
# Step 1: Load Dataset
# -----------------------------------

df = pd.read_csv("salaries.csv")

print("Dataset:")
print(df.head())


# -----------------------------------
# Step 2: Separate Inputs and Target
# -----------------------------------

inputs = df.drop('salary_more_then_100k', axis='columns')
target = df['salary_more_then_100k']


# -----------------------------------
# Step 3: Label Encoding
# -----------------------------------

le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

inputs['company_n'] = le_company.fit_transform(inputs['company'])
inputs['job_n'] = le_job.fit_transform(inputs['job'])
inputs['degree_n'] = le_degree.fit_transform(inputs['degree'])


# -----------------------------------
# Step 4: Remove Original Text Columns
# -----------------------------------

inputs_n = inputs.drop(
    ['company', 'job', 'degree'],
    axis='columns'
)

print("\nNumerical Input Data:")
print(inputs_n)


# -----------------------------------
# Step 5: Create Decision Tree
# -----------------------------------

model = tree.DecisionTreeClassifier()

model.fit(inputs_n, target)


# -----------------------------------
# Step 6: Accuracy
# -----------------------------------

accuracy = model.score(inputs_n, target)

print("\nModel Accuracy:")
print(accuracy)

print("Accuracy Percentage:", accuracy * 100, "%")


# -----------------------------------
# Step 7: Prediction 1
# Google + Computer Engineer + Bachelors
# -----------------------------------

# Check actual encoded values
print("\nCompany classes:", list(le_company.classes_))
print("Job classes:", list(le_job.classes_))
print("Degree classes:", list(le_degree.classes_))

google_n = le_company.transform(['Google'])[0]
ce_n = le_job.transform(['Computer Engineer'])[0]
bachelors_n = le_degree.transform(['Bachelors'])[0]
masters_n = le_degree.transform(['Masters'])[0]

prediction1 = model.predict([[google_n, ce_n, bachelors_n]])

print("\nGoogle + Computer Engineer + Bachelors:")
print(prediction1)

if prediction1[0] == 1:
    print("Salary is MORE than 100K")
else:
    print("Salary is NOT more than 100K")


# -----------------------------------
# Step 8: Prediction 2
# Google + Computer Engineer + Masters
# -----------------------------------

prediction2 = model.predict([[google_n, ce_n, masters_n]])

print("\nGoogle + Computer Engineer + Masters:")
print(prediction2)

if prediction2[0] == 1:
    print("Salary is MORE than 100K")
else:
    print("Salary is NOT more than 100K")


# -----------------------------------
# Step 9: Decision Tree Graph
# -----------------------------------

plt.figure(figsize=(16, 10))

tree.plot_tree(
    model,
    feature_names=['Company', 'Job', 'Degree'],
    class_names=['Salary <= 100K', 'Salary > 100K'],
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("Decision Tree - Salary Prediction", fontsize=14)
plt.tight_layout()
plt.savefig("decision_tree_salary.png", dpi=150)
plt.show()

print("\nDecision Tree graph saved as 'decision_tree_salary.png'")


# -----------------------------------
# Step 10: Prediction Comparison Bar Chart
# -----------------------------------

labels = ['Bachelors', 'Masters']
predictions = [prediction1[0], prediction2[0]]
colors = ['#e74c3c' if p == 0 else '#2ecc71' for p in predictions]

plt.figure(figsize=(8, 5))
bars = plt.bar(labels, predictions, color=colors, edgecolor='black', width=0.5)

for bar, pred in zip(bars, predictions):
    label = "Yes (> 100K)" if pred == 1 else "No (<= 100K)"
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.05,
        label,
        ha='center',
        fontsize=12,
        fontweight='bold'
    )

plt.title("Google + Computer Engineer\nSalary > 100K Prediction", fontsize=14)
plt.ylabel("Prediction (0 = No, 1 = Yes)")
plt.ylim(-0.2, 1.5)
plt.tight_layout()
plt.savefig("salary_prediction_comparison.png", dpi=150)
plt.show()

print("Comparison chart saved as 'salary_prediction_comparison.png'")
