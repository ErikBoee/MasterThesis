import numpy as np
import os

for j in range(6):
    for lamda in [100, 1000, 10000, 100000]:
        for i in range(5):
            filename = "j_" + str(j) + "_lambda_" + str(lamda) + "_i_" + str(int(i)) + ".npy"
            path = "Runs_finished/Mobius_test/" + filename
            if os.path.exists(path):
                problem_dictionary = np.load(path,
                                             allow_pickle=True).item()
                print("j =", j, "i = ", i, "Lambda = ", lamda)
                print("--------------------------")
                print(problem_dictionary)
                print("--------------------------")
