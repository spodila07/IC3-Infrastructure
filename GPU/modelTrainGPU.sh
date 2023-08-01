#!/bin/bash                     # Shebang - specifies bash as the script interpreter

#SBATCH --job-name=pytorch_test # Assigns a name to the job for identification
#SBATCH --output=pytorch.out    # Specifies the file to save job's standard output
#SBATCH --error=pytorch.err     # Specifies the file to save job's standard error
#SBATCH --mail-type=ALL         # Enables all types of email notifications
#SBATCH --mail-user=email@ufl.edu # Specifies the email address to receive notifications
#SBATCH --nodes=1               # Allocates one node for the job
#SBATCH --ntasks=8              # Allocates 8 tasks/processes for the job
#SBATCH --cpus-per-task=1       # Assigns 1 CPU for each task
#SBATCH --ntasks-per-node=8     # Allocates 8 tasks per node
#SBATCH --distribution=cyclic:cyclic # Sets task distribution to cyclic across nodes and sockets
#SBATCH --mem-per-cpu=1000mb    # Allocates 1000mb of memory for each CPU
#SBATCH --partition=gpu         # Specifies the GPU partition
#SBATCH --gpus=a100:1           # Assigns 1 A100 GPU for the job
#SBATCH --time=00:30:00         # Sets a time limit of 30 minutes for the job
#SBATCH --qos=uf-iccc -b


module purge                    # Removes all currently loaded modules to ensure a clean environment

module load python/3.8          # Loads the Python version 3.8 module
module load cuda/11.0           # Loads the CUDA version 11.0 module for GPU computing

python -m venv myenv            # Creates a Python virtual environment in a directory "myenv"
source myenv/bin/activate       # Activates the created virtual environment

pip install torch torchvision matplotlib
# Uses pip to install the necessary Python packages in the virtual environment

srun python modelTrainGPU.py               # Executes the Python script named "modelTrainGPU.py" using the srun command

