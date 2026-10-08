import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree

# Step 2: Load Titanic dataset
# Note: Since direct download from Kaggle requires authentication, we'll use a public mirror
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Display first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Step 3: Select useful parameters
inputs = df[['Pclass', 'Sex', 'Age', 'Fare']].copy()

# Target variable
target = df['Survived']

# Step 4: Convert Sex into numbers
# male = 1, female = 0
le = LabelEncoder()
inputs['Sex'] = le.fit_transform(inputs['Sex'])

# Step 5: Handle missing Age values
inputs['Age'] = inputs['Age'].fillna(inputs['Age'].mean())

# Step 6: Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    inputs, target, test_size=0.2, random_state=42
)

# Step 7: Create Decision Tree model
model = DecisionTreeClassifier(
    criterion='gini', max_depth=4, random_state=42
)

# Step 8: Train the model
model.fit(X_train, y_train)

# Step 9: Make predictions
predictions = model.predict(X_test)

# Step 10: Calculate model score
score = model.score(X_test, y_test)
print("\nModel Score:", score)
print("Accuracy:", accuracy_score(y_test, predictions))

# Step 11: Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Step 12: Classification Report
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Step 13: Save the Decision Tree Plot
plt.figure(figsize=(20, 10))
plot_tree(
    model, 
    feature_names=['Pclass', 'Sex', 'Age', 'Fare'], 
    class_names=['Not Survived', 'Survived'], 
    filled=True
)
plt.savefig('decision_tree.png')
print("\nDecision tree plot saved as 'decision_tree.png'.")
