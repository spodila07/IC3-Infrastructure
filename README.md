
# IC3 System Infrastructure




## Contents

- HiperGator Overview
- SLURM Overview
- SLRUM Commands 
- SLRUM Scripts


## HiperGator Overview

HiPerGator is the high-performance computing (HPC) system at the University of Florida. As one of the most powerful supercomputers in academia, it is an invaluable asset for research across various scientific domains. HiPerGator uses SLURM as its job scheduler, making it directly compatible with the types of jobs and resource allocation


## SLURM Overview

The Simple Linux Utility for Resource Management (SLURM) is a free and open-source job scheduler used for allocating computational resources in a high-performance computing (HPC) environment. It provides a robust framework to execute and manage diverse computational tasks across one or more clusters of machines, facilitating both research and industrial applications that require significant computational power.

#### Resource Allocation
SLURM handles the distribution of computational resources such as CPU cores, memory, and GPUs among various queued tasks. Resource requirements can be specified using flags like `--nodes`, `--ntasks`, `--cpus-per-task`, and `--mem-per-cpu`. The Quality of Service (`--qos`) flag allows for further fine-grained control.

#### Job Types
#####  Array Jobs : 
Useful for hyperparameter tuning, array jobs can run the same script with different input parameters.


###### GPU Jobs : 
Specialized jobs that utilize GPU resources, ideal for deep learning and other complex computations.

###### Multiprocess Jobs : 
These jobs run on a single node but utilize multiple CPU cores. Often used for parallelizing tasks within a single machine.

###### Multithread Jobs : 
Similar to multiprocess jobs but are more suitable for tasks that share the same memory space.

###### MultiNode Jobs : 
These jobs run across multiple nodes, each possibly having multiple tasks. Ideal for distributed computing tasks.

###### Single Thread Jobs : 
Simple jobs that run on a single thread, typically used for tasks that do not benefit from parallelization.

###### Spark Jobs : 
Specialized jobs designed for big data processing using Apache Spark.

#### Environment Setup
SLURM scripts usually begin by setting up the computing environment, including loading necessary modules and setting up virtual Python environments.

#### Notifications
SLURM allows for various types of notifications such as job start, end, and failure, which can be sent to a specified email address.

#### Script Execution
The main script or program to be run is usually invoked at the end of the SLURM script. This could be a machine learning training script, a data analysis script, or any other computational task.

#### Logging and Output
Output and error files can be specified to capture the standard output and standard error streams. This helps in debugging and monitoring.

#### Flexibility and Customization
SLURM is extremely flexible, allowing you to define very customized resource requirements, job arrays for parameter sweeps, and specialized environment setups. 

By efficiently handling all these aspects, SLURM enables researchers and engineers to focus more on the computational tasks themselves, rather than the intricacies of resource allocation and job management.
## SLURM Commands

- `sbatch`: Submit a batch job to the SLURM scheduler.

   Example: `sbatch my_job_script.sh`

- `squeue`: View the job queue and status of submitted jobs.

   Example: `squeue -u your_username`

- `scancel`: Cancel a running or pending job.

   Example: `scancel job_id`

- `sinfo`: Display information about the nodes in the cluster.

   Example: `sinfo`

- `scontrol`: Control aspects of the SLURM scheduler and running jobs.

   - View job details: `scontrol show job job_id`
   - View node details: `scontrol show node node_name`
   - Hold a job: `scontrol hold job job_id`
   - Release a job from hold: `scontrol release job job_id`

- `sacct`: View accounting information and job statistics.

   Example: `sacct -u your_username`
- `srun`: Run a command within a SLURM allocation (typically used in job scripts).

   Example: `srun my_program`

- `scontrol update`: Modify the properties of a running job, such as the number of nodes or CPUs.

   Example: `scontrol update job_id NumNodes=3`

- `sacct` command in SLURM displays the accounting data related to jobs, job steps, and other activities on the cluster. It can show detailed information about resource usage, job states, time, and more. Various filters and options can be applied to narrow down the results.

  Example: `sacct -u your_username`

- `sreport` command in SLURM allows you to generate various reports using the accounting data stored within the system. It provides insights into the cluster's utilization, user activities, and more. You can customize the report by using different options and filters

  Example: `sreport user top Usage Start=lastmonth`
## SLURM Scripts

### Array Job Script

#### Basic Settings

- `--job-name=array_job`: Specifies the name of the job.
- `--output=array_job_%A_%a.out`: The stdout file name pattern. `%A` and `%a` represent the job ID and array index, respectively.
- `--error=array_job_%A_%a.err`: The stderr file name pattern.
- `--array=1-3`: The array index range, specifying that three jobs (1, 2, 3) will be run.
  
#### Resource Requests

- `--ntasks=1`: Number of tasks for each job array element.
- `--cpus-per-task=1`: Number of CPU cores per task.
- `--mem-per-cpu=1G`: Memory requested per CPU core.
- `--time=01:00:00`: Time limit for the job execution.
- `--qos=uf-iccc-b`: Quality of Service, specific to your cluster settings.

#### Environment and Modules

- The script loads the `python` module.
- You can optionally activate a Python environment by uncommenting the relevant line (`# conda activate <env_name>`).

#### Hyperparameters


The script defines two lists of hyperparameters:
- `n_estimators_list`: Contains `[100, 200, 50]`.
- `max_depth_list`: Contains `[5, 10, 3]`.

The specific hyperparameter values for each job array element are determined using the `SLURM_ARRAY_TASK_ID`.

#### Execution

Finally, the Python script `modelTrainArray.py` is run with the selected `n_estimators` and `max_depth` as arguments.

To execute the script, save it in a file (e.g., `submit_job.sh`) and run: `sbatch submit_job.sh`

### GPU Script

#### Basic Settings

- `--job-name=pytorch_test`: Specifies the name of the job for easier identification.
- `--output=pytorch.out`: Redirects the job's standard output to a file named `pytorch.out`.
- `--error=pytorch.err`: Redirects the job's standard error to a file named `pytorch.err`.
- `--mail-type=ALL`: Enables all types of email notifications.
- `--mail-user=email@ufl.edu`: Specifies the email address to receive notifications.

#### Resource Requests

- `--nodes=1`: Requests one node for the job.
- `--ntasks=8`: Specifies that 8 tasks/processes will be used for this job.
- `--cpus-per-task=1`: Allocates 1 CPU core per task.
- `--ntasks-per-node=8`: Allocates 8 tasks for each node.
- `--distribution=cyclic:cyclic`: Sets the task distribution to be cyclic across nodes and sockets.
- `--mem-per-cpu=1000mb`: Allocates 1000 megabytes of memory for each CPU core.
- `--partition=gpu`: Specifies that the job will run on the GPU partition.
- `--gpus=a100:1`: Requests one A100 GPU for the job.
- `--time=00:30:00`: Sets a time limit of 30 minutes for the job execution.

#### Environment and Modules

- `module purge`: Removes all currently loaded modules to start with a clean environment.
- `module load python/3.8`: Loads Python version 3.8.
- `module load cuda/11.0`: Loads CUDA version 11.0 for GPU computing.

#### Virtual Environment and Packages

- The script creates a Python virtual environment in a directory called `myenv`.
- It activates the virtual environment and installs essential Python packages (`torch`, `torchvision`, `matplotlib`) using `pip`.

#### Execution

The Python script `ex.py` is executed using the `srun` command.

To submit the job, save the script in a file (e.g., `submit_pytorch_job.sh`) and run: `sbatch submit_pytorch_job.sh`

### MultiProcess Script

#### Basic Settings

- `--job-name=modelTrainMultiProcess_job_test`: Names the job for easier identification.
- `--mail-type=ALL`: Enables all types of email notifications.
- `--mail-user=<email_address>`: Specifies the email address for notifications.

#### Resource Requests

- `--nodes=1`: Requests a single node for the job.
- `--ntasks=1`: Specifies that one task will be used for this job.
- `--cpus-per-task=4`: Allocates 4 CPU cores per task.
- `--mem=2G`: Sets a memory limit of 2 GB for the job.
- `--time=00:05:00`: Specifies a time limit of 5 minutes for the job.
- `--output=modelTrainMultiProcess_test_%j.out`: Redirects the job's standard output to a file.
- `--qos=uf-iccc-b`: Sets the Quality of Service (adjust as needed).

#### Environment and System Information

- The script prints out various details like the current working directory, hostname, date/time, SLURM version, and Python version for better reproducibility.
- It also gathers information about the CPU architecture and, if applicable, GPU details.
- It lists the number of CPU sockets, cores, and threads per core.

#### Python Script Execution

- The Python module is loaded, and the script `modelTrainMultiProcess.py` is executed.
- Date and time are printed at the end to mark the job completion.

#### Execution

Save this Bash script into a file, e.g., `submit_modelTrainMultiProcess.sh`. To submit the job, run: `sbatch submit_modelTrainMultiProcess.sh`

### Multithread Script

#### Basic Settings

- `--job-name=modelTrainMultiThread_job_test`: Names the job for easier identification.
- `--mail-type=ALL`: Enables all types of email notifications.
- `--mail-user=<email_address>`: Specifies the email address for notifications.

#### Resource Requests

- `--nodes=1`: Requests a single node for the job.
- `--ntasks=1`: Specifies that one task will be used for this job.
- `--cpus-per-task=4`: Allocates 4 CPU cores per task.
- `--mem=2G`: Sets a memory limit of 2 GB for the job.
- `--time=00:05:00`: Specifies a time limit of 5 minutes for the job.
- `--output=modelTrainMultiThread_test_%j.out`: Redirects the job's standard output to a file.
- `--qos=uf-iccc-b`: Sets the Quality of Service (adjust as needed).

#### Environment Variable for OpenMP

- `export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK`: Sets the number of OpenMP threads equal to the number of allocated CPU cores.

#### Environment and System Information

- The script prints out various details like the current working directory, hostname, date/time, SLURM version, and Python version for better reproducibility.
- It also gathers information about the CPU architecture and, if applicable, GPU details.
- It lists the number of CPU sockets, cores, and threads per core.

#### Python Script Execution

- The Python module is loaded, and the script `modelTrainMultiThread.py` is executed.
- Date and time are printed at the end to mark the job completion.

#### How to Run

Save this Bash script into a file, e.g., `submit_modelTrainMultiThread.sh`. To submit the job, run: `sbatch submit_modelTrainMultiThread.sh`

### MultipleNode

#### Basic Settings

- `--job-name=modelTrainMultipleNodesJob`: Specifies the name of the job for easier identification.
- `--output=modelTrainMultipleNodes_output.log`: Redirects the job's standard output to a file named `modelTrainMultipleNodes_output.log`.
- `--qos=uf-iccc-b`: Specifies the Quality of Service (QoS) level for the job.

#### Resource Requests

- `--nodes=3`: Requests 3 nodes for the job.
- `--ntasks-per-node=1`: Specifies that one task will be run on each node.
- `--mem-per-cpu=1GB`: Allocates 1GB of memory per CPU core.
- `--time=00:30:00`: Sets a time limit of 30 minutes for the job to run.
- `--partition=hpg-default`: Specifies that the job will run on the 'hpg-default' partition of the cluster.

#### Environment and Modules

- `module purge`: Removes all currently loaded modules to start with a clean environment.
- `module load python/3.10`: Loads Python version 3.10.

#### Virtual Environment and Packages

- The script creates a Python virtual environment in a directory called `/blue/uf-iccc/spodila/myenv`.
- It activates the virtual environment and installs essential Python packages (`dask`, `distributed`, `scikit-learn`, `joblib`, `dask-jobqueue`, `pandas`) using `pip`.

#### Execution

The Python script `modelTrainMultipleNode.py` is executed directly. To submit the job, save the script in a file (e.g., `submit_dask_multi_node_job.sh`) and run:
`sbatch submit_dask_multi_node_job.sh`


### Single Threaded Script

#### Basic Settings

- `--job-name=modelTrainSingleThread_job_test`: This assigns a specific name to the job for easier identification in the job queue.
- `--mail-type=ALL`: This will send email notifications at the beginning, end, and in case of an error in the job.
- `--mail-user=<email_address>`: The email address to which notifications will be sent.
- `--output=modelTrainSingleThread_test_%j.out`: Redirects both standard output and standard error messages into a log file.
- `--qos=uf-iccc-b`: Quality of Service parameter. Adjust according to your needs.

#### Resource Requests

- `--nodes=1`: Requests a single node for running the job.
- `--ntasks=1`: Specifies the execution of one task, which in this case is the running of a Python script.
- `--cpus-per-task=1`: Allocates one CPU core for that task.
- `--mem=2G`: Requests 2GB of RAM.
- `--time=00:05:00`: Limits the total runtime to 5 minutes.

##### Environment and Modules

- Prints the current working directory, hostname, and current date and time.
- Outputs the versions of SLURM and Python being used.
- Outputs the architecture of the CPU.

#### Virtual Environment and Packages

- The script does not explicitly create or activate a virtual environment. If needed, this would be the place to do it.

#### Execution

- The script prints out various pieces of system information, which can be useful for debugging.
- Then, the Python script `modelTrainSingleThread.py` is executed.
 - Once the job is finished, the script prints the current date and time again to indicate the end of the job.

To submit the job, save this script in a file and use the `sbatch` command: `sbatch submit_single_thread_job.sh`

### Spark Job Script

#### Basic Settings

- `--job-name=IrisSparkJob`: Assigns a specific name to the job to make it easier to identify in the job queue.
- `--output=iris_spark_job_%j.out`: Standard output is directed to a file with the job's ID.
- `--error=iris_spark_job_%j.err`: Standard error is directed to a separate file with the job's ID.
- `--qos=uf-iccc-b`: Specifies the Quality of Service parameter. This can be adjusted according to your cluster's policies.

#### Resource Requests

- `--nodes=1`: Requests a single node for running the job.
- `--ntasks-per-node=4`: Specifies the execution of 4 tasks (or processes) on the single node.
- `--mem=4G`: Requests 4GB of memory.
- `--time=00:30:00`: Sets a time limit of 30 minutes for the job to complete.

#### Environment and Modules

- `module load java/11.0.1`: This loads the Java module, version 11.0.1.
- `export JAVA_HOME=/apps/java/jdk-11.0.1`: This sets the JAVA_HOME environment variable, based on your specific Java installation.
- `module load python`: Loads the Python module for executing Python scripts.

> Note: If your cluster provides Spark as a module, you can uncomment `module load spark` to load the Spark module.

#### Virtual Environment and Packages

This script doesn't set up a Python virtual environment or install Python packages. You would add those here if needed for your specific Spark job.

#### Execution

- The environment modules for Java and Python are loaded.
- The `JAVA_HOME` environment variable is set.
- Finally, the script runs your Spark Python script `modelTrainSpark.py`.

To submit this job, save the script and use the `sbatch` command:
`sbatch submit_spark_job.sh`

## Acknowledgements

 - [Hipergator New User Training](https://help.rc.ufl.edu/doc/New_user_training)
 - [UF Sample SLURM Scripts](https://help.rc.ufl.edu/doc/Sample_SLURM_Scripts)
 - [UF SLURM Commands](https://help.rc.ufl.edu/doc/SLURM_Commands)
- [SLURM Documentation](https://help.rc.ufl.edu/doc/Sample_SLURM_Scripts)
