#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-imf
#SBATCH --time=5-00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=2000
#SBATCH --job-name="job_1"
#SBATCH --output=job_1.out
#SBATCH --mail-user=erikob@stud.ntnu.no
#SBATCH --mail-type=ALL

module load GCCcore/8.3.0
module load Python/3.7.4-GCCcore-8.3.0
source ../../optimization_idun_4/bin/activate

python3 solve_problem.py
