#!/bin/bash -l
#SBATCH --job-name=r_serial
#SBATCH --account=project_2012477
#SBATCH --output=output_%j.txt
#SBATCH --error=errors_%j.txt
#SBATCH --partition=test
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1000

# Load necessary modules if applicable (e.g., Python, NumPy)
source /projappl/project_2012477/implementation/.venv/bin/activate

cd /projappl/project_2012477/implementation/adm-emo/adm_emo/NSGAII/combined_indicator/k=3/

srun $IRACE_HOME/irace