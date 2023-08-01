#!/bin/bash
#SBATCH --job-name=dask-worker
#SBATCH --nodes=5                # Specify the number of nodes
#SBATCH --ntasks=5               # Allocate 5 tasks/processes for the job (one for each node)
#SBATCH --cpus-per-task=2        # Assign 3 CPUs for each task
#SBATCH --mem-per-cpu=1G         # Allocate 2GB of memory for each CPU
#SBATCH --time=01:00:00          # Set a time limit of 1 hour for the job
#SBATCH --output=dask-worker.o%j # Specify the file to save job's standard output. %j is replaced by the job ID
#SBATCH --partition=hpg-default  # Specify the partition/queue where the job will be submitted
#SBATCH --qos=uf-iccc-b
module purge                     # Removes all currently loaded modules to ensure a clean environment
module load python/3.10          # Loads the Python version 3.10 module
python3 -m venv env              # Creates a Python virtual environment in a directory "env"
source env/bin/activate          # Activates the created virtual environment
pip install dask distributed scikit-learn joblib dask-jobqueue pandas # Uses pip to install the required Python packages in the virtual environment
python ex.py                     # Runs the Python script named "ex.py"

