# IC3-Infrastructure
HiPerGator is a high-performance computing (HPC) system used for research at the University of Florida. It's one of the most powerful university supercomputers in the United States. "HiPerGator" is a play on words, combining "High-Performance" with the University of Florida's mascot, the Gator.

Key Features

Scalability: HiPerGator is designed to be highly scalable, accommodating many users and complex computational tasks. Its modular architecture allows expansion with additional CPU and GPU clusters to meet growing demand.
Parallel Computing: The system is optimized for parallel computing. With multi-core CPUs and many-node architecture, it excels at tasks requiring simultaneous processing and can perform complex simulations and data analysis at high speeds.
Resource Allocation: Resources like CPU time, memory, and storage are dynamically allocated to users based on their research needs and the system's limitations. This ensures maximum utilization of the HPC resources.
Software Libraries: HiPerGator comes preloaded with a broad range of software libraries and tools optimized for high-performance computing tasks, saving researchers time in setting up their computational environments.
Data Storage: The system provides various options for data storage, including high-speed SSDs for temporary data and more extensive, slower drives for long-term storage. Some configurations may also offer in-memory data storage for high-speed data access.
Node Diversity: HiPerGator offers a range of compute nodes tailored to different research needs, from general-purpose CPU nodes to specialized GPU-accelerated nodes for tasks that require high parallelism.
CPUs: Utilizes latest-generation multi-core processors for general-purpose computing tasks.
RAM: Equipped with high-capacity RAM modules for data-intensive tasks like genome sequencing or fluid dynamics simulations.
GPUs: Includes nodes with high-performance GPUs for tasks that are highly parallelizable, such as deep learning and molecular dynamics simulations.
Accelerators: We may offer FPGA or hardware accelerators for specialized computational needs.
Fault Tolerance and Redundancy: The system architecture is designed with fault tolerance and redundancy. Critical components like power supplies and network switches are often duplicated to ensure uninterrupted operation.
High-Speed Interconnect: Utilizes low-latency, high-bandwidth network technology to facilitate fast data transfer between nodes, crucial for parallel computing tasks and rapid I/O operations.
Resource Scheduling: Incorporates advanced job scheduling algorithms like fair-share and priority queuing to ensure efficient distribution of computational tasks among all active users.
Security Features: Access is controlled through secure SSH protocols, firewalls, and virtual private networks to ensure data integrity and confidentiality. Additional layers of security may include two-factor authentication.
Environment Modules: Though you wanted to exclude modules, they are vital in managing software environments, allowing users to switch easily between software and library versions without conflicts.
Batch Processing: Automated batch job submissions are facilitated through job schedulers like Slurm or Torque, enabling users to execute unattended tasks and manage workloads efficiently.
Essential Steps to Access HiPerGator

GatorLink Account: Ensure you have a GatorLink account, as it is usually required for authentication.
SSH Client: Ensure you have an SSH client installed on your computer.
Windows users can use software like PuTTY.
macOS and Linux users can use the built-in terminal.
SSH Command: Open your SSH client and enter the following command to connect to HiPerGator.
 ssh [username]@hpg.rc.ufl.edu
Replace [username] with your GatorLink username.
Authentication: You may be prompted to enter your password and go through two-factor authentication.
Optional: VPN If you are accessing HiPerGator from off-campus, you should connect via the University of Florida's VPN service.

VPN Software: Download and install the VPN client the University of Florida recommended.
Connect to VPN: Open the VPN client, enter your credentials, and connect.
SSH Access: Once the VPN connection is established, you can access HiPerGator via SSH, as mentioned above.
SLURM provides a clearer understanding of its power and flexibility as an HPC resource manager and job scheduler. Below are some of the technical features and capabilities that make SLURM an industry-standard choice:

Technical Features of SLURM

Node and Socket Allocation: SLURM can allocate resources at various granularities, ranging from entire nodes to individual CPU sockets, cores, and threads.
Gang Scheduling and Preemption: SLURM supports "gang scheduling," where tasks are scheduled to run simultaneously. Additionally, higher-priority jobs can preempt lower-priority ones using mechanisms like "checkpoint/restart" or "queue."
Memory Pooling: SLURM can pool the memory resources of multiple nodes, allowing jobs to access more memory than what is available on a single node.
Array Jobs: SLURM allows for creating an array of jobs, making it easier to manage and schedule many similar jobs.
Resource Reservation: SLURM supports advanced reservation of resources. You can schedule a set of nodes to be available for your job at a specific time.
Dependency Scheduling: Jobs can be set to run based on the completion, failure, or success of other jobs, creating complex workflow dependencies.
Power Management: SLURM can integrate with power management solutions to dynamically power up or down nodes based on workload requirements, improving energy efficiency.
Plugin Architecture: SLURM is designed with a modular plugin architecture, allowing custom functionality to be added easily. This includes new scheduling algorithms, prologue/epilogue scripts, and SPANK plugins for user-defined tasks.
Networking Topology Awareness: SLURM knows the network topology and can optimize job placement for better data locality, reducing the latency and increasing the bandwidth for MPI jobs.
Fair-share Scheduling: It employs a multi-factor priority plugin that considers historical usage, ensuring that resources are distributed fairly among users and groups over time.
Job Accounting and Reporting: SLURM can be configured to use various accounting backends like MySQL or HDF5, allowing for detailed usage tracking and reporting.
Authentication and Security: Supports multiple forms of secure authentication, including Munge and Kerberos, and can be configured to enforce various security policies.
Advanced Usage

Job Script Directives: Users can specify resource requirements and other job options directly in their batch script using #SBATCH directives. For example, #SBATCH --nodes=2 will request two nodes.
Job Steps: SLURM allows jobs to be broken down into "job steps," which are individual portions of a job that can be scheduled and managed separately. This is useful for running MPI programs.
Run, batch, and cancel are essential command-line utilities for interacting with SLURM. Run is used for submitting jobs for immediate execution, batch is used for batch job submission, and science is used to cancel pending or running jobs.
Monitoring Tools: SLURM provides a range of monitoring tools like queue, info, and control for real-time inspection of job queues, cluster status, and detailed job information.
SLURM Commands

SLURM (Simple Linux Utility for Resource Management) is a workload manager used in high-performance computing (HPC) environments to schedule and manage jobs on a cluster. Here are some basic SLURM commands:

sbatch: Submit a batch job to the SLURM scheduler.Example: sbatch my_job_script.sh
squeue: View the job queue and status of submitted jobs.Example: squeue -u your_username
scancel: Cancel a running or pending job.Example: scancel job_id
sinfo: Display information about the nodes in the cluster.Example: sinfo
scontrol: Control aspects of the SLURM scheduler and running jobs.
View job details: scontrol show job job_id
View node details: scontrol show node node_name
Hold a job: scontrol hold job job_id
Release a job from hold: scontrol release job job_id
sacct: View accounting information and job statistics.Example: sacct -u your_username
srun: Run a command within a SLURM allocation (typically used in job scripts).Example: srun my_program
scontrol update: Modify the properties of a running job, such as the number of nodes or CPUs.Example: scontrol update job_id NumNodes=3
'sacct' command in SLURM displays the accounting data related to jobs, job steps, and other activities on the cluster. It can show detailed information about resource usage, job states, time, and more. Various filters and options can be applied to narrow down the results.
Example: sacct -u your_username 10.'sreport' command in SLURM allows you to generate various reports using the accounting data stored within the system. It provides insights into the cluster's utilization, user activities, and more. You can customize the report by using different options and filters Example: sreport user top Usage Start=lastmonth

SLURM Scripts:

Array Job Script:

Script Purpose: The script runs a SLURM array job to execute a Python model training script (modelTrainArray.py). It performs hyperparameter tuning by running the script with different combinations of hyperparameters (n_estimators and max_depth).

Script Components:

Job Configuration
--job-name: Sets the name of the job as array_job.
--output/--error: Defines the output and error log files, using job and array indices.
--array=1-3: Specifies that three array tasks will be created.
--ntasks, --cpus-per-task, --mem-per-cpu: Resource allocations for each array task.
--time: Time limit for each array task.
--qos: Specifies the Quality of Service setting.
Environment Setup
Loads the python module.
(Optional) Activates a Python environment using conda.
Hyperparameter List
n_estimators_list and max_depth_list hold possible hyperparameters.
Parameter Retrieval
Retrieves hyperparameters for each specific array task based on SLURM_ARRAY_TASK_ID.
Job Execution
Calls the Python script modelTrainArray.py with the chosen hyperparameters as arguments.
GPU Job Script :

Objective: The script schedules a job to run a PyTorch script (ex.py) on a GPU-enabled node, utilizing specific CPU and memory allocations. It also sets up a Python virtual environment and installs necessary Python packages.

Major Components:

Job Identification
--job-name=pytorch_test: Labels the job for easy identification.
--output/--error: Defines where to save the standard output and error.
Email Notifications
--mail-type=ALL: Enables all types of email notifications.
--mail-user: Defines the email address to receive these notifications.
Resource Allocation
--nodes, --ntasks, --cpus-per-task, --mem-per-cpu: Specify resources like number of nodes, tasks, CPUs, and memory.
--partition=gpu, --gpus=a100:1: Chooses the GPU partition and allocates one A100 GPU.
--time=00:30:00: Sets the maximum time limit for the job as 30 minutes.
--distribution=cyclic:cyclic: Cyclic distribution of tasks across nodes and CPU sockets.
Environment Setup
module purge: Clears all preloaded modules for a clean slate.
module load python/3.8 and module load cuda/11.0: Loads Python and CUDA modules.
Sets up a Python virtual environment and activates it.
Installs the required Python packages (torch, torchvision, matplotlib).
Script Execution
srun python ex.py: Executes the Python script ex.py using SLURM's srun.
Multiprocess Job Script:

Objective: The script schedules a SLURM job to execute a Python script designed for multi-process model training (modelTrainMultiProcess.py). The job runs on a single node, utilizing specified CPU cores and memory.

Major Components:

Job Identification
--job-name: Assigns the name modelTrainMultiProcess_job_test for easy identification.
--output: Defines the output log file with job-specific information.
Email Notifications
--mail-type=ALL: Enables all types of email notifications.
--mail-user: Specifies the email address to receive notifications.
Resource Allocation
--nodes=1, --ntasks=1, --cpus-per-task=4, --mem=2G: Specifies the resource requirements including number of nodes, tasks, CPU cores, and memory.
--time=00:05:00: Sets the maximum time limit for the job as 5 minutes.
--qos=uf-iccc-b: Sets the Quality of Service.
System Information Retrieval
Prints details like working directory, hostname, date and time, environment details, and CPU and GPU info.
Script Execution
Loads the Python module and runs the modelTrainMultiProcess.py Python script.
Job Completion Mark
Prints the end time for the job.
Multithread Job Script

Objective: The script schedules a SLURM job to execute a Python script designed for multi-threaded model training (modelTrainMultiThread.py). It uses a single node, with specified CPU cores and memory allocations.

Major Components:

Job Identification
--job-name: Sets the name of the job as modelTrainMultiThread_job_test.
--output: Specifies the output log file, tagging it with the job ID.
Email Notifications
--mail-type=ALL: Enables all types of email notifications.
--mail-user: Specifies the email address to receive notifications.
Resource Allocation
--nodes=1, --ntasks=1, --cpus-per-task=4, --mem=2G: Allocates resources including the number of nodes, tasks, CPU cores, and memory.
--time=00:05:00: Sets a maximum time limit of 5 minutes for the job.
--qos=uf-iccc-b: Sets the Quality of Service.
OMP Threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK: Sets the number of OpenMP threads to the number of CPU cores allocated by SLURM.
System Information
Prints working directory, hostname, date, time, software versions, CPU, and GPU information.
Script Execution
Loads Python and runs the modelTrainMultiThread.py Python script.
Job Completion
Prints the end date and time to mark the end of the job
MultiNode Job Script:

Objective: The script schedules a SLURM job to execute a Python script (modelTrainMultipleNode.py) designed for distributed model training across multiple nodes.

Major Components:

Shell and Job Identification
#!/bin/bash: Specifies that the Bash shell is used to interpret the script.
--job-name: Names the job as modelTrainMultipleNodesJob.
--output: Directs the job's standard output to a log file.
Email Notifications
This feature is not included in the script. To enable, you can use --mail-type and --mail-user flags.
Resource Allocation
--nodes=3, --ntasks-per-node=1, --mem-per-cpu=1GB: Requests 3 nodes with 1 task per node and 1GB of memory per CPU.
--time=00:30:00: Sets a 30-minute time limit for the job.
--partition=hpg-default: Submits the job to the "hpg-default" partition.
--qos=uf-iccc-b: Specifies the Quality of Service.
Environment Setup
module purge: Clears all previously loaded modules.
module load python/3.10: Loads Python 3.10.
Sets up a virtual Python environment and activates it.
Installs required Python packages using pip.
Script Execution
Executes the modelTrainMultipleNode.py Python script.
Single Thread Job Script

Objective: This script is intended to schedule a SLURM job that will run a Python script (modelTrainSingleThread.py) for model training on a single thread.

Major Components:

  Shell and Job Identification
#!/bin/bash: Specifies that the Bash shell will interpret the script.
--job-name: Assigns the name modelTrainSingleThread_job_test to the job.
--output: Directs the job's standard output to a log file.
  Email Notifications
--mail-type=ALL, --mail-user=<email_address>: Enables all types of email notifications and specifies the recipient email address.
  Resource Allocation
--nodes=1, --ntasks=1, --cpus-per-task=1, --mem=2G: Requests one node with one task and one CPU core. Allocates 2GB of memory.
--time=00:05:00: Sets a 5-minute time limit for the job.
--qos=uf-iccc-b: Specifies the Quality of Service.
  Job Metadata
Various echo commands provide information like the current working directory, hostname, current date/time, and machine architecture.
  Script Execution
Executes the Python script (modelTrainSingleThread.py) using the default Python interpreter.
Spark Job Script:

Objective: The script schedules a SLURM job to run a Spark job, utilizing Apache Spark to process the Iris dataset. The Python script (modelTrainSpark.py) used for this Spark job is assumed to be in the directory from where the SLURM script is run.

Major Components:

  Shell and Job Identification
#!/bin/bash: Specifies that the Bash shell will interpret the script.
--job-name: Assigns the name IrisSparkJob to the job.
--output, --error: Directs the job's standard output and standard error to log files.
  Email Notifications
Not specified in this script. Add --mail-type and --mail-user if you need email notifications.
  Resource Allocation
--nodes=1, --ntasks-per-node=4, --mem=4G: Requests one node with four tasks and 4GB of memory.
--time=00:30:00: Sets a 30-minute time limit for the job.
--qos=uf-iccc-b: Specifies the Quality of Service.
  Environment Setup
Loads Java module and sets the JAVA_HOME variable.
Optionally loads Spark module, depending on your cluster configuration.
  Job Execution
Executes the Python script (modelTrainSpark.py) using the Python interpreter.
