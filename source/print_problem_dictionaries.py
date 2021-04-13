import numpy as np
import os

for j in range(5):
    for lamda in [100, 1000, 10000, 100000]:
        for i in np.linspace(0, 5, 6):
            filename = "j_" + str(j) + "_lambda_" + str(lamda) + "_i_" + str(int(i)) + ".npy"
            path = "Runs_finished/Star_prob_5_noise_0_0_beta_0_5_no_angles_8_lambda_100_1000/" + filename
            if os.path.exists(path):
                problem_dictionary = np.load(path,
                                             allow_pickle=True).item()
                print("j =", j, "i = ", i, "Lambda = ", lamda)
                print("--------------------------")
                print(problem_dictionary)
                print("--------------------------")
