
# Import required libraries
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Step 2: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Step 3: Create the classifiers
gaussian_model = GaussianNB()
multinomial_model = MultinomialNB()

# Step 4: Train both models
gaussian_model.fit(X_train, y_train)
multinomial_model.fit(X_train, y_train)

# Step 5: Predict test samples
gaussian_pred = gaussian_model.predict(X_test)
multinomial_pred = multinomial_model.predict(X_test)

# Step 6: Evaluate and compare accuracy
gaussian_acc = accuracy_score(y_test, gaussian_pred) * 100
multinomial_acc = accuracy_score(y_test, multinomial_pred) * 100

print("Gaussian Naive Bayes Accuracy:", gaussian_acc, "%")
print("Multinomial Naive Bayes Accuracy:", multinomial_acc, "%")

# Step 7: Display detailed evaluation
print("\nGaussian Classification Report:")
print(classification_report(y_test, gaussian_pred))

print("\nMultinomial Classification Report:")
print(classification_report(y_test, multinomial_pred))

# Step 8: Compare actual and predicted labels
print("\nActual labels:     ", y_test[:10])
print("Gaussian predictions:", gaussian_pred[:10])
print("Multinomial predictions:", multinomial_pred[:10])