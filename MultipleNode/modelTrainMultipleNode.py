import joblib
import logging
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets

# Logging configuration
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
file_handler = logging.FileHandler('output.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(file_handler)

def train_model(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    logging.info(f"Model accuracy: {accuracy}")

    # Save the trained model
    joblib.dump(clf, 'trained_model.joblib')
    logging.info("Trained model saved to 'trained_model.joblib'")

    return clf

def evaluate_model(clf, X_test, y_test):
    accuracy = clf.score(X_test, y_test)
    logging.info(f"Model evaluation accuracy: {accuracy}")

if __name__ == "__main__":
    try:
        cluster = SLURMCluster(
            cores=2,
            memory='2GB',
            queue='hpg-default',
            local_directory='/blue/uf-iccc/spodila/SLURM_Models/multipleNodes'
        )
        cluster.scale(jobs=5)  # Request 5 nodes
        client = Client(cluster)
        client.wait_for_workers(n_workers=5)  # Wait for workers

        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

        clf = train_model(X_train, X_test, y_train, y_test)
        evaluate_model(clf, X_test, y_test)

    except Exception as e:
        logging.error("Encountered error while running script: ", exc_info=True)

    finally:
        if 'client' in locals():  # Ensure client exists before closing
            client.close()
        logging.info("Script completed.")

