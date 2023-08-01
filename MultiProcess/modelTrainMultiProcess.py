import numpy as np
import pickle
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
import multiprocessing as mp
import logging

# Function to preprocess a chunk of data
def preprocess_data(data_chunk):
    # Unpack the data_chunk into features and labels
    X, y = data_chunk

    # Scale features to have zero mean and unit variance
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Return the preprocessed features and labels
    return X, y

# Function to train and evaluate a model on a chunk of data
def train_and_evaluate_model(data_chunk):
    # Unpack the data_chunk into features and labels
    X, y = data_chunk

    # Define a logistic regression model
    model = LogisticRegression()

    # Define the grid of hyperparameters for grid search
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}

    # Define grid search with 5-fold cross-validation
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=1)

    # Train the model using grid search
    grid_search.fit(X, y)

    # Return the best model found by grid search
    return grid_search.best_estimator_

def main():
    # Set up logging to write messages to a log file
    logging.basicConfig(filename='logfile.log', level=logging.INFO)

    # Load iris dataset
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # Divide the data into 4 stratified chunks
    sss = StratifiedShuffleSplit(n_splits=4, test_size=0.5, random_state=0)
    indices = list(sss.split(X, y))

    # Create chunks of data for training models
    chunks = [(X[train_idx], y[train_idx]) for train_idx, test_idx in indices]

    # Create a pool of 4 worker processes
    with mp.Pool(processes=4) as pool:
        # Use the pool to apply the preprocess_data function to each chunk
        # Each call to preprocess_data will be executed in a separate process
        chunks = pool.map(preprocess_data, chunks)

    # Create another pool of 4 worker processes
    with mp.Pool(processes=4) as pool:
        # Use the pool to apply the train_and_evaluate_model function to each chunk
        # Each call to train_and_evaluate_model will be executed in a separate process
        models = pool.map(train_and_evaluate_model, chunks)

    # To aggregate the individual models into a single model, we'll use model averaging
    # Model averaging is a straightforward method that involves simply averaging the model coefficients

    # Create an empty model to hold the averaged coefficients and intercept
    averaged_model = LogisticRegression()

    # For the coefficients, calculate the mean across all models' coefficients
    # We use axis=0 to calculate the mean of each feature's coefficient across all models
    averaged_model.coef_ = np.mean([model.coef_ for model in models], axis=0)

    # For the intercept, calculate the mean across all models' intercepts
    # Since the intercept is a single value for each model, no axis is needed for np.mean
    averaged_model.intercept_ = np.mean([model.intercept_ for model in models], axis=0)

    # Log the details of the averaged model
    logging.info(f"Averaged Model:\n{averaged_model}")

    # Save the averaged model to a file
    with open('averaged_model.pkl', 'wb') as f:
        pickle.dump(averaged_model, f)

# Run the main function when this script is executed
if __name__ == "__main__":
    main()

