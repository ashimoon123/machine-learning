import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = load_iris()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Create a logistic regression model for multiclass classification
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Predict the categories for the test set
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the Logistic Regression model: {accuracy:.2f}")

# Predict a few samples from the test dataset
print("\nPredictions for first 5 samples in test set:")
for i in range(5):
    predicted_class = iris.target_names[y_pred[i]]
    actual_class = iris.target_names[y_test[i]]
    print(f"Sample {i+1} Features: {X_test[i]}")
    print(f"  Predicted: {predicted_class}, Actual: {actual_class}")
