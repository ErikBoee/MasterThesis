#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-imf
#SBATCH --time=1-00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=12000
#SBATCH --job-name="testidun"
#SBATCH --output=testidun.out
#SBATCH --mail-user=erikob@stud.ntnu.no
#SBATCH --mail-type=ALL

module load GCCcore/.9.3.0
module load Python/3.8.2

source ../optimization/bin/activate

python3 visualize_problem.py