# Ensemble Model with Hyperparameter Tuning for Iris Dataset
# This script demonstrates building an ensemble model using Gradient Boosting Classifier with hyperparameter tuning
# on the Iris dataset. The ensemble model combines multiple Gradient Boosting classifiers trained on different
# subsets of the data using multithreading to speed up the process. The best hyperparameters for each classifier
# are determined through grid search with cross-validation. The final ensemble model aggregates predictions from
# individual models to make the final prediction for unseen data.
# The script also evaluates the ensemble model's performance on a test set and saves the trained model for future use.
# Additionally, it shows how to load the saved ensemble model and make predictions on new data.

# Import necessary libraries
import numpy as np
from scipy.stats import mode
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.utils import shuffle
import joblib

# Define hyperparameters for grid search
param_grid = {
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
}

# Custom ensemble model class
# The EnsembleModel class combines the predictions of individual models to make the final prediction.
# It uses the mode function from scipy.stats to find the most common prediction among the individual models.
# This class will be used to create the final ensemble model.
class EnsembleModel:
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        # Predict the class for each model and take the most common prediction
        predictions = np.array([model.predict(X) for model in self.models])
        final_predictions = mode(predictions, axis=0)[0][0]
        return final_predictions

# Load and preprocess the dataset
def load_and_preprocess_data():
    # Loading the iris dataset
    iris = load_iris()
    # Shuffle the dataset to ensure randomness
    X, y = shuffle(iris.data, iris.target, random_state=42)
    # Split the dataset into a train and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Split the training set into eight parts for multithreading
    Xs = np.array_split(X_train, 8)
    ys = np.array_split(y_train, 8)
    # Return a list of tuples, each containing a split of X_train and y_train, and also return the test set
    return list(zip(Xs, ys)), X_test, y_test

# Train the model with hyperparameter tuning
def train_and_tune_model(data):
    # Separate data into features and labels
    X_train, y_train = data
    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    # Define Gradient Boosting Classifier
    model = GradientBoostingClassifier()
    # Define grid search with cross-validation
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
    # Fit grid search on training data
    grid_search.fit(X_train, y_train)
    # Return the best Gradient Boosting Classifier model
    return grid_search.best_estimator_

# Evaluate the ensemble model's performance on the test set
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
    print('Classification Report:\n', classification_report(y_test, y_pred))

def main():
    # Create a ThreadPoolExecutor with a fixed number of worker threads
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Load and preprocess the data
        data_splits, X_test, y_test = load_and_preprocess_data()
        # Submit each data split for training and hyperparameter tuning
        futures_train = [executor.submit(train_and_tune_model, data) for data in data_splits]

        trained_models = []
        # Collect the trained models as they become ready
        for future in as_completed(futures_train):
            # Add the model to the list of trained models
            trained_models.append(future.result())

    # Create an ensemble model from the trained models
    ensemble_model = EnsembleModel(trained_models)

    # Evaluate ensemble model on the test set
    print("Ensemble Model Evaluation:")
    evaluate_model(ensemble_model, X_test, y_test)

    # Save the ensemble model for future use
    joblib.dump(ensemble_model, 'ensemble_model.pkl')

    # To use the saved model for predictions, load it like this:
    # loaded_model = joblib.load('ensemble_model.pkl')
    # y_pred = loaded_model.predict(X_test)

if __name__ == "__main__":
    main()

