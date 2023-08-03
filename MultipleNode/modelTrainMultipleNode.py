from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from dask import delayed
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
import logging
import joblib

# Logging configuration
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
file_handler = logging.FileHandler('output.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(file_handler)

def train_model(X_train, y_train, n_jobs):
    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0, n_jobs=n_jobs)
    clf.fit(X_train, y_train)
    return clf

if __name__ == "__main__":
    try:
        # Start the Dask cluster across multiple nodes
        cluster = SLURMCluster(cores=1, memory='1GB', queue='hpg-default')
        cluster.scale(jobs=3)  # Request 3 nodes
        client = Client(cluster)
        client.wait_for_workers(3)

        # Generate synthetic dataset
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

        # Number of jobs is set to match the total number of cores available
        n_jobs = 3

        # Calculate data distribution based on the number of workers
        total_workers = 3
        chunk_size = len(X_train) // total_workers
        remainder = len(X_train) % total_workers

        # Train the model using all workers (nodes) in parallel
        training_tasks = []
        start = 0
        for worker_index in range(total_workers):
            end = start + chunk_size + (1 if worker_index < remainder else 0)
            X_train_worker = X_train[start:end]
            y_train_worker = y_train[start:end]

            # Delay the training task
            training_task = delayed(train_model)(X_train_worker, y_train_worker, n_jobs)
            training_tasks.append(training_task)

            start = end

        # Compute the delayed objects
        trained_models = client.compute(training_tasks)

        # Gather the results
        trained_models_local = client.gather(trained_models)

        # Combine individual RandomForest models into a single ensemble model using Voting
        combined_model = VotingClassifier(estimators=[(f'model_{i}', model) for i, model in enumerate(trained_models_local)], voting='hard')
        combined_model = combined_model.fit(X_train, y_train)  # Refit on the full dataset
        joblib.dump(combined_model, 'combined_model.pkl')

        # Evaluate the combined model on the test data
        accuracy = combined_model.score(X_test, y_test)
        logging.info(f"Combined model accuracy: {accuracy}")

    except Exception as e:
        logging.error("Encountered error while running script: ", exc_info=True)

    finally:
        # Close the Dask client and cluster
        client.close()
        cluster.close()
        logging.info("Script completed.")
