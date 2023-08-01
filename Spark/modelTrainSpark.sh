#!/bin/bash
#SBATCH --job-name=IrisSparkJob           # Assigns a name to the job
#SBATCH --nodes=1                         # Request one node for the job
#SBATCH --ntasks-per-node=4               # Request 4 tasks (processes) per node
#SBATCH --time=00:30:00                   # Set a time limit of 30 minutes for the job
#SBATCH --mem=4G                          # Request 4GB of memory for the job
#SBATCH --output=iris_spark_job_%j.out    # Direct standard output to a file named iris_spark_job_JOBID.out
#SBATCH --error=iris_spark_job_%j.err     # Direct standard error to a file named iris_spark_job_JOBID.err
#SBATCH --qos=uf-iccc-b               # Quality of Service (adjust as needed)


# Load the Java module
module load java/11.0.1                   # Load Java module version 11.0.1

# Set the JAVA_HOME variable (modify this path according to your cluster's Java installation)
export JAVA_HOME=/apps/java/jdk-11.0.1  # Set JAVA_HOME environment variable

# Load the Spark module or set up the Spark environment
# module load spark # Uncomment if your cluster provides Spark as a module; Load Spark module

# Path to your Spark script
module load python
python modelTrainSpark.py                      # Path to the Python script to run



