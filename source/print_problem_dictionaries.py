import numpy as np
import os

for j in range(10):
    for lamda in [100, 1000, 10000, 100000]:
        for i in range(6):
            filename = "j_" + str(j) + "_lambda_" + str(lamda) + "_i_" + str(int(i)) + ".npy"
            path = "Experiments_finished/Experiment_5/Experiment_5_noise_0_15_beta_1_6_no_angles_16_lambda_100_1000000/" + filename
            if os.path.exists(path):
                problem_dictionary = np.load(path,
                                             allow_pickle=True).item()
                print("j =", j, "i = ", i, "Lambda = ", lamda)
                print("--------------------------")
                print(problem_dictionary)
                print("--------------------------")
