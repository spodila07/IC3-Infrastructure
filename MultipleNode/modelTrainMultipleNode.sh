#!/bin/bash                                                     # Use the Bash shell to interpret the script
#SBATCH --job-name=modelTrainMultipleNodesJob                   # Set job name to "ex3"
#SBATCH --nodes=3                                               # Request 3 nodes
#SBATCH --ntasks-per-node=1                                     # Run one task per node
#SBATCH --mem-per-cpu=1GB                                       # Allocate 1GB of memory per CPU
#SBATCH --time=00:30:00                                         # Limit job run time to 30 minutes
#SBATCH --partition=hpg-default                                 # Submit job to the "hpg-default" partition
#SBATCH --output=modelTrainMultipleNodes_output.log             # Send output to "ex3_output.log" file
#SBATCH --qos=uf-iccc-b                                         # Assign Quality of Service (QoS) level

module purge                                                    # Remove all loaded modules to start with a clean slate
module load python/3.10                                         # Load Python 3.10

python3 -m venv /blue/uf-iccc/spodila/myenv  # Create virtual environment
source /blue/uf-iccc/spodila/myenv/bin/activate # Activate virtual environment

pip install dask distributed scikit-learn joblib dask-jobqueue pandas # Install required Python packages

python modelTrainMultipleNode.py                                # Run Python script for model training
