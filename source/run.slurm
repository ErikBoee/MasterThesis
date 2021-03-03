#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-imf
#SBATCH --time=3-00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=12000
#SBATCH --job-name="Circle_not_fine_grid"
#SBATCH --output=Circle_not.out
#SBATCH --mail-user=erikob@stud.ntnu.no
#SBATCH --mail-type=ALL

module load GCCcore/.7.3.0
module load Python/3.6.6
source ../../optimization/bin/activate

python3 idun_run.py
