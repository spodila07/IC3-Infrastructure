# Importing required libraries and modules
from dask.distributed import Client        # Client module for connecting to a Dask cluster
from dask_jobqueue import SLURMCluster     # SLURMCluster module to interface with the SLURM job scheduler
from dask import delayed                   # 'delayed' function for parallelizing Python functions
from sklearn.datasets import make_classification  # For generating a synthetic classification dataset
from sklearn.ensemble import RandomForestClassifier, VotingClassifier # RandomForest for modeling and VotingClassifier for ensembling
from sklearn.model_selection import train_test_split # For splitting data into training and testing sets
import logging                             # For logging information and errors
import joblib                              # For saving the model to disk

# Setting up logging with a specific format, logging both to console and a file
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
file_handler = logging.FileHandler('output.log') # Writing log messages to 'output.log'
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(file_handler)

# Define function to train a RandomForest model
def train_model(X_train, y_train, n_jobs):
    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0, n_jobs=n_jobs)
    clf.fit(X_train, y_train) # Train the RandomForest model on the provided training data
    return clf # Return the trained model

# Main block of code
if __name__ == "__main__":
    try:
        # Create a Dask cluster on SLURM with specific resource allocation
        cluster = SLURMCluster(cores=1, memory='1GB', queue='hpg-default')
        cluster.scale(jobs=3) # Request 3 worker nodes from the SLURM job scheduler
        client = Client(cluster) # Initialize a Dask client to interact with the cluster
        client.wait_for_workers(3) # Wait until 3 workers are ready

        # Create a synthetic dataset for classification, with 1000 samples and 20 features
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        # Split dataset into 60% training and 40% testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

        # Define the number of jobs (parallel tasks) for training; matching the number of worker nodes
        n_jobs = 3

        # Divide the training data equally for parallel processing
        total_workers = 3
        chunk_size = len(X_train) // total_workers # Size of each chunk
        remainder = len(X_train) % total_workers # Any remaining samples after dividing

        # Initialize a list to store delayed tasks for training
        training_tasks = []
        start = 0 # Starting index for slicing
        for worker_index in range(total_workers):
            # Compute ending index for slicing, accommodating any remainder
            end = start + chunk_size + (1 if worker_index < remainder else 0)
            # Slice the data for the current worker
            X_train_worker = X_train[start:end]
            y_train_worker = y_train[start:end]
            # Create a delayed task for training the model on the sliced data
            training_task = delayed(train_model)(X_train_worker, y_train_worker, n_jobs)
            training_tasks.append(training_task)
            start = end # Update the starting index for the next iteration

        # Compute the delayed objects, initiating the parallel training
        trained_models = client.compute(training_tasks)

        # Retrieve the trained models from the workers
        trained_models_local = client.gather(trained_models)

        # Ensemble the individual RandomForest models using majority voting
        combined_model = VotingClassifier(estimators=[(f'model_{i}', model) for i, model in enumerate(trained_models_local)], voting='hard')
        combined_model = combined_model.fit(X_train, y_train) # Refit on the full dataset
        joblib.dump(combined_model, 'combined_model.pkl') # Save to disk

        # Evaluate the combined model on the testing data
        accuracy = combined_model.score(X_test, y_test)
        logging.info(f"Combined model accuracy: {accuracy}") # Log the accuracy result

    except Exception as e:
        # If any exceptions are raised, log the full error information
        logging.error("Encountered error while running script: ", exc_info=True)

    finally:
        # Close the client and cluster, releasing the allocated resources
        client.close()
        cluster.close()
        logging.info("Script completed.") # Log the completion of the script
