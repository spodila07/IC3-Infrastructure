#This Python script demonstrates the process of training a Random Forest Classifier on the Iris dataset.
# It allows the user to specify the number of trees (n_estimators) and the maximum depth of the trees (max_depth)
# as command-line arguments. The script loads the Iris dataset, splits it into training and testing sets,
# trains the Random Forest model with the specified hyperparameters, and evaluates the model's accuracy on the test data.

import sys
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_and_split_data():
    """
    Load the Iris dataset and split it into training and testing sets.

    Returns:
        X_train (numpy array): Features of the training set.
        X_test (numpy array): Features of the testing set.
        y_train (numpy array): Labels of the training set.
        y_test (numpy array): Labels of the testing set.
    """
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, n_estimators, max_depth):
    """
    Train a Random Forest Classifier on the given data with specified hyperparameters.

    Args:
        X_train (numpy array): Features of the training set.
        y_train (numpy array): Labels of the training set.
        n_estimators (int): Number of trees in the forest.
        max_depth (int): Maximum depth of the trees.

    Returns:
        model (RandomForestClassifier): Trained Random Forest model.
    """
    # Create a model with the given hyperparameters
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    # Train the model on the training data
    model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test data.

    Args:
        model (RandomForestClassifier): Trained Random Forest model.
        X_test (numpy array): Features of the testing set.
        y_test (numpy array): Labels of the testing set.

    Returns:
        accuracy (float): Accuracy of the model on the test data.
    """
    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

if __name__ == "__main__":
    # Validate command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python train_model.py <n_estimators> <max_depth>")
        sys.exit(1)

    try:
        n_estimators = int(sys.argv[1])
        max_depth = int(sys.argv[2])
    except ValueError:
        print("Error: Invalid input for n_estimators or max_depth. Please provide integers.")
        sys.exit(1)

    X_train, X_test, y_train, y_test = load_and_split_data()
    model = train_model(X_train, y_train, n_estimators, max_depth)
    accuracy = evaluate_model(model, X_test, y_test)

    # Print the accuracy and model details
    print(f"Model with n_estimators={n_estimators}, max_depth={max_depth}, Accuracy={accuracy:.2f}")
    print("Model Details:")
    print(model)

