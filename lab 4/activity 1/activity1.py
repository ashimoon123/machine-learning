import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn import metrics


# ==========================================
# 1. LOAD DATASET
# ==========================================

col_names = [
    'pregnant',
    'glucose',
    'bp',
    'skin',
    'insulin',
    'bmi',
    'pedigree',
    'age',
    'label'
]

pima = pd.read_csv(
    "diabetes.csv",
    header=None,
    names=col_names
)

print("Dataset Loaded Successfully!")
print(pima.head())


# ==========================================
# 2. FEATURE SELECTION
# ==========================================

feature_cols = [
    'pregnant',
    'insulin',
    'bmi',
    'age',
    'glucose',
    'bp',
    'pedigree'
]

X = pima[feature_cols]
y = pima['label']


# ==========================================
# 3. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=1
)


# ==========================================
# 4. ORIGINAL DECISION TREE
# ==========================================

model1 = DecisionTreeClassifier()

model1.fit(X_train, y_train)

y_pred1 = model1.predict(X_test)

accuracy1 = metrics.accuracy_score(
    y_test,
    y_pred1
)

print("\nOriginal Decision Tree Accuracy:")
print(accuracy1)


# ==========================================
# 5. SHOW ORIGINAL TREE
# ==========================================

plt.figure(figsize=(20, 10))

plot_tree(
    model1,
    feature_names=feature_cols,
    class_names=['No Diabetes', 'Diabetes'],
    filled=True,
    rounded=True
)

plt.title("Original Decision Tree - Diabetes Prediction")

plt.show()


# ==========================================
# 6. PRUNED / OPTIMIZED TREE
# ==========================================

model2 = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3
)

model2.fit(X_train, y_train)

y_pred2 = model2.predict(X_test)

accuracy2 = metrics.accuracy_score(
    y_test,
    y_pred2
)

print("\nOptimized Decision Tree Accuracy:")
print(accuracy2)


# ==========================================
# 7. SHOW PRUNED TREE
# ==========================================

plt.figure(figsize=(18, 8))

plot_tree(
    model2,
    feature_names=feature_cols,
    class_names=['No Diabetes', 'Diabetes'],
    filled=True,
    rounded=True
)

plt.title("Pruned Decision Tree - Diabetes Prediction")

plt.show()


# ==========================================
# 8. ACCURACY GRAPH
# ==========================================

models = [
    "Original Tree",
    "Optimized Tree"
]

accuracies = [
    accuracy1 * 100,
    accuracy2 * 100
]

plt.figure(figsize=(8, 5))

bars = plt.bar(
    models,
    accuracies
)

plt.ylabel("Accuracy (%)")
plt.xlabel("Model")
plt.title("Decision Tree Model Accuracy")

plt.ylim(0, 100)

# Show accuracy value above bars
for bar, value in zip(bars, accuracies):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.2f}%",
        ha='center'
    )

plt.show()
