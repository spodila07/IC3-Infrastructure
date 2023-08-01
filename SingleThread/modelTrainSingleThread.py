import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import joblib
import logging
import matplotlib.pyplot as plt

# Enable logging to both console and file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.StreamHandler(),  # Print logs to console
    logging.FileHandler('output.log')  # Save logs to file
])

def load_dataset():
    # Load the Iris dataset, a built-in dataset in scikit-learn for demonstration
    iris = load_iris()
    X = iris.data  # Feature data
    y = iris.target  # Target labels
    return X, y

def main():
    # Get the script directory
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Load the dataset
    X, y = load_dataset()

    # List of numeric and categorical features (For Iris dataset all are numeric)
    numeric_features = list(range(X.shape[1]))
    categorical_features = []

    # Numeric transformer for imputing and scaling
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Fill missing values with median
        ('scaler', RobustScaler())  # Scale features using robust scaler to handle outliers
    ])

    # Categorical transformer for imputing and encoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Fill missing values with 'missing'
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
    ])

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),  # Apply numeric transformer to numeric features
            ('cat', categorical_transformer, categorical_features)  # Apply categorical transformer to categorical features
        ])

    # Feature Selection
    feature_selection = SelectFromModel(RandomForestClassifier(n_estimators=100))  # Use Random Forest for feature selection

    # Define the model
    model = LogisticRegression(max_iter=1000)  # Logistic Regression as the main model

    # Create a pipeline with preprocessing, feature selection, and the model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', feature_selection),
        ('model', model)
    ])

    # Hyperparameters grid for the model
    param_grid = {
        'model__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],  # Solver algorithms for Logistic Regression
        'model__C': np.logspace(-4, 4, 20),  # Regularization parameter for Logistic Regression
    }

    # Use GridSearchCV for hyperparameter tuning with 5-fold stratified cross-validation
    grid_search_cv = GridSearchCV(pipeline, param_grid, cv=StratifiedKFold(n_splits=5), n_jobs=-1)
    grid_search_cv.fit(X, y)

    # Best hyperparameters
	
    logging.info(f"Best Parameters: {grid_search_cv.best_params_}")
    # Best score (mean cross-validation accuracy)
    logging.info(f"Best Score: {grid_search_cv.best_score_}")

    # Save the trained model with best hyperparameters
    joblib.dump(grid_search_cv.best_estimator_, os.path.join(script_dir, 'model.pkl'))

    # Evaluate the model with cross-validation on the entire dataset
    # cross_val_score to perform cross-validation again with the best estimator (model with the best hyperparameters).
    # The cross_val_score function returns an array of accuracy scores for each fold of the cross-validation.
    # The mean of these scores (np.mean(scores)) represents the overall performance of the model on the entire dataset.
    scores = cross_val_score(grid_search_cv.best_estimator_, X, y, cv=StratifiedKFold(n_splits=5))
    logging.info(f"Cross Validation Score: {np.mean(scores)}")

    # Extract feature importance if the model is Logistic Regression
    if isinstance(grid_search_cv.best_estimator_.named_steps['model'], LogisticRegression):
        if grid_search_cv.best_estimator_.named_steps['model'].coef_ is not None:
            importance = grid_search_cv.best_estimator_.named_steps['model'].coef_[0]
            feature_importance = pd.DataFrame(list(zip(load_iris().feature_names, importance)),
                                              columns=['Feature', 'Importance'])
            logging.info(f"Feature Importance: {feature_importance}")

if __name__ == "__main__":
    main()

