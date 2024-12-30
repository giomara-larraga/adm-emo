#!/bin/bash -l
#SBATCH --job-name=rvea_5
#SBATCH --account=project_2012477
#SBATCH --output=output_%j.txt
#SBATCH --error=errors_%j.txt
#SBATCH --partition=medium
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem-per-cpu=1000

# Load necessary modules if applicable (e.g., Python, NumPy)
source /projappl/project_2012477/implementation/.venv/bin/activate

cd /projappl/project_2012477/implementation/adm-emo/adm_emo/RVEA/combined_indicator/k=5/

srun irace --parallel $SLURM_CPUS_PER_TASK