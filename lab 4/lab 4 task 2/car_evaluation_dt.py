import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    # URL of the dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
    
    # Column names based on the dataset description
    col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    
    print("Downloading and loading the dataset...")
    df = pd.read_csv(url, names=col_names)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    
    # Define ordinal mappings for features
    mappings = {
        'buying': {'low': 1, 'med': 2, 'high': 3, 'vhigh': 4},
        'maint': {'low': 1, 'med': 2, 'high': 3, 'vhigh': 4},
        'doors': {'2': 2, '3': 3, '4': 4, '5more': 5},
        'persons': {'2': 2, '4': 4, 'more': 5},
        'lug_boot': {'small': 1, 'med': 2, 'big': 3},
        'safety': {'low': 1, 'med': 2, 'high': 3}
    }
    
    # We will predict the 'class' column (overall car acceptability). 
    # If you specifically wanted to predict the 'safety' column instead of the 'class' column,
    # you can change the target variable below.
    X = df.drop('class', axis=1)
    
    # Apply ordinal mapping to the features
    for col in mappings:
        if col in X.columns:
            X[col] = X[col].map(mappings[col])
            
    y = df['class']
    
    print("\nSplitting dataset into training and testing sets (70% train, 30% test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print("Training the Decision Tree Classifier...")
    # Initialize the Decision Tree Classifier
    clf = DecisionTreeClassifier(criterion='gini', random_state=42)
    clf.fit(X_train, y_train)
    
    print("Evaluating the model on the test set...")
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
