#!/bin/bash
#SBATCH --job-name=array_job           # Name of the job
#SBATCH --output=array_job_%A_%a.out   # Output file for job stdout
#SBATCH --error=array_job_%A_%a.err    # Error file for job stderr
#SBATCH --array=1-3                   # Number of hyperparameter configurations (adjust this based on your hyperparams_list)
#SBATCH --ntasks=1                    # Number of tasks to run (1 in this case as we're running a single Python script)
#SBATCH --cpus-per-task=1             # Number of CPU cores per task
#SBATCH --mem-per-cpu=1G              # Memory per CPU core
#SBATCH --time=01:00:00               # Time limit for job execution (adjust as needed)
#SBATCH --qos=uf-iccc-b               # Quality of Service (adjust as needed)

# Load the required modules (if needed)
module load python

# Activate the Python environment (if needed)
# conda activate <env_name>

# Run the Python script with different hyperparameters
# Define lists of hyperparameters
n_estimators_list=(100 200 50)
max_depth_list=(5 10 3)

# Get the specific hyperparameter values for this array task
n_estimators=${n_estimators_list[$SLURM_ARRAY_TASK_ID - 1]}
max_depth=${max_depth_list[$SLURM_ARRAY_TASK_ID - 1]}

# Call the Python script and pass the hyperparameters as arguments
python modelTrainArray.py $n_estimators $max_depth

