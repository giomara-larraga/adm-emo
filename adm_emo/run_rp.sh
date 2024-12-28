#!/bin/bash

#SBATCH --job-name=gen_ref_points        # Job name
#SBATCH --account=project_2012477
#SBATCH --output=test_output_%j.log          # Output log file (%j is the job ID)
#SBATCH --error=test_error_%j.log            # Error log file
#SBATCH --time=05:00:00                      # Maximum runtime (HH:MM:SS)
#SBATCH --ntasks=1                           # Number of tasks (usually 1 for srun and Python scripts)
#SBATCH --partition=small                        # Replace with the partition name (e.g., general, short)
#SBATCH --mem-per-cpu=16G                            # Memory per node (adjust as needed)

# Load necessary modules if applicable (e.g., Python, NumPy)
source /projappl/project_2012477/implementation/.venv/bin/activate

# Navigate to the directory containing your scripts
cd /projappl/project_2012477/implementation/adm-emo/adm_emo/

# Run the Python script with srun
srun python adm_get_reference_points.py