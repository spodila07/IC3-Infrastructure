#!/bin/bash
#SBATCH --job-name=modelTrainMultiThread_job_test     # Job name
#SBATCH --mail-type=ALL                # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=<email_address>    # Where to send mail
#SBATCH --nodes=1                      # Use one node
#SBATCH --ntasks=1                     # Run a single task
#SBATCH --cpus-per-task=4              # Use 1 core` (change as needed)
#SBATCH --mem=2G                       # Memory limit (change as needed)
#SBATCH --time=00:05:00                # Time limit (hrs:min:sec)
#SBATCH --output=modelTrainMultiThread_test_%j.out    # Standard output and error log
#SBATCH --qos=uf-iccc-b               # Quality of Service (adjust as needed)


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK  # Set number of OMP threads to the number of cores allocated by SLURM

# Print the current working directory, hostname, and current date and time
echo "Current working directory: $(pwd)"
echo "Hostname: $(hostname)"
echo "Date: $(date)"

# Record the exact versions of software/modules used for better reproducibility
echo "Environment details:"
echo "SLURM version: $(sbatch --version | head -n 1)"
echo "Python version: $(python --version 2>&1)"

# Determine the CPU architecture
echo "CPU Architecture:"
lscpu | grep "Architecture:"

# Check for GPU usage (if applicable)
echo "GPU Information:"
nvidia-smi -L 2>/dev/null || echo "No GPUs found."

# Record the number of sockets, cores, and threads used
echo "Number of Sockets: $(lscpu | grep 'Socket(s)' | awk '{print $2}')"
echo "Number of Cores per Socket: $(lscpu | grep 'Core(s) per socket' | awk '{print $4}')"
echo "Number of Threads per Core: $(lscpu | grep 'Thread(s) per core' | awk '{print $4}')"

# Run the Python script
# Replace 'modelTrainSingleThread.py' with the actual filename of your Python script
# Make sure the Python script is in the same directory as this SLURM script or provide the correct path
module load python
python modelTrainMultiThread.py

# Print the current date and time again to mark the end of the job
echo "Job finished at: $(date)"
