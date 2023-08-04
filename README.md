# IC3-Infrastructure

This Reposiroty talks about the implementation of model training under different instances.

**1. Model Train Using Array Job**

At a high level, the Python script performs the following tasks:
Loading the Iris Dataset: It reads the Iris dataset, which consists of features like petal length, petal width, sepal length, and sepal width, along with corresponding species labels.
Splitting the Data: It divides the dataset into two parts: a training set to train the model and a testing set to evaluate its performance.
Training a Random Forest Model: It creates and trains a Random Forest classifier using the training data. The number of trees in the forest and their maximum depth are specified through command-line arguments.
Evaluating the Model: It tests the trained Random Forest model on the testing set to see how well it predicts the species of unseen iris flowers.
Reporting the Results: The script prints the model's accuracy on the testing set along with details of the trained Random Forest model.
The script encapsulates a complete machine learning workflow, from data loading to model training, evaluation, and reporting, specifically focusing on the Random Forest algorithm applied to the Iris dataset.

**Implementation Details:**

The code uses NumPy arrays to manage the Iris dataset throughout the entire process:

Loading Data: The Iris dataset is loaded as two NumPy arrays: a 2D array for the features (sepal and petal dimensions) and a 1D array for the labels (species).
Data Splitting: train_test_split() divides the data into training and testing sets, returning 2D arrays for the training/testing features and 1D arrays for the corresponding labels.
Model Training: The RandomForestClassifier's fit() method uses the training feature and label arrays to train the model.
Model Evaluation: The predict() method uses the testing feature array to generate predictions (1D array), which are compared with the proper labels to calculate accuracy.
NumPy arrays are central to storing and handling the dataset, facilitating the training and evaluation of a Random Forest model on the Iris data.

3. Model Tran Using GPU's
4. Model Train Using Multi threading
5. Model Train Using Multi Processing
6. Model Train using Multiple Nodes
7. Model Train in Single Threaded Fashion
8. Model Train using Spark
  
