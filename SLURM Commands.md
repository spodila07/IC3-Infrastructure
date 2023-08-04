SLURM (Simple Linux Utility for Resource Management) is a workload manager used in high-performance computing (HPC) environments to schedule and manage jobs on a cluster. Here are some basic SLURM commands:

1. `sbatch`: Submit a batch job to the SLURM scheduler.

   Example: `sbatch my_job_script.sh`

2. `squeue`: View the job queue and status of submitted jobs.

   Example: `squeue -u your_username`

3. `scancel`: Cancel a running or pending job.

   Example: `scancel job_id`

4. `sinfo`: Display information about the nodes in the cluster.

   Example: `sinfo`

5. `scontrol`: Control aspects of the SLURM scheduler and running jobs.

   - View job details: `scontrol show job job_id`
   - View node details: `scontrol show node node_name`
   - Hold a job: `scontrol hold job job_id`
   - Release a job from hold: `scontrol release job job_id`

6. `sacct`: View accounting information and job statistics.

   Example: `sacct -u your_username`

7. `srun`: Run a command within a SLURM allocation (typically used in job scripts).

   Example: `srun my_program`

8. `scontrol update`: Modify the properties of a running job, such as the number of nodes or CPUs.

   Example: `scontrol update job_id NumNodes=3`

9. 'sacct' command in SLURM displays the accounting data related to jobs, job steps, and other activities on the cluster. It can show detailed information about resource usage, job states, time, and more. Various filters and options can be applied to narrow down the results.

  Example: `sacct -u your_username`

10.'sreport' command in SLURM allows you to generate various reports using the accounting data stored within the system. It provides insights into the cluster's utilization, user activities, and more. You can customize the report by using different options and filters

  Example: `sreport user top Usage Start=lastmonth`
