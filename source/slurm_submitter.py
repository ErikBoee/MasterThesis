import os

job_directory = os.getcwd()

i = 0
experiment_number = 2
for filename in os.listdir("Experiments/Experiment_" + str(experiment_number)):
    job_file = "run_%s.slurm" % str(i)
    if ("beta_0_04" in filename or "beta_0_16" in filename or
            "beta_0_05" in filename or "beta_0_08" in filename):
        if not os.path.exists(job_file):
            with open(job_file, 'a+') as fh:
                fh.writelines("#!/bin/sh\n")
                fh.writelines("#SBATCH --partition=CPUQ\n")
                fh.writelines("#SBATCH --account=share-ie-imf\n")
                fh.writelines("#SBATCH --time=5-00:10:00\n")
                fh.writelines("#SBATCH --nodes=1\n")
                fh.writelines("#SBATCH --ntasks-per-node=1\n")
                fh.writelines("#SBATCH --mem=2000\n")
                fh.writelines('#SBATCH --job-name="job_' + str(i) + '_' + str(experiment_number) + '"\n')
                fh.writelines("#SBATCH --output=job_" + str(i) + '_' + str(experiment_number) + ".out\n")
                fh.writelines("#SBATCH --mail-user=erikob@stud.ntnu.no\n")
                fh.writelines("#SBATCH --mail-type=ALL\n\n")

                fh.writelines("module load GCCcore/8.3.0\n")
                fh.writelines("module load Python/3.7.4-GCCcore-8.3.0\n\n")
                fh.writelines("source ../../optimization_idun_4/bin/activate\n\n")

                fh.writelines("python3 idun_run.py" + " " + filename)
        os.system("chmod u+x %s" % job_file)
        os.system("sbatch %s" % job_file)
        i += 1
