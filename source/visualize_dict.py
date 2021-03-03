import numpy as np
import os

for j in range(10):
    for lamda in [100, 1000, 10000, 100000]:
        for i in range(11):
            filename = "j_" + str(j) + "_lambda_" + str(lamda) + "_i_" + str(i) + ".npy"
            if os.path.exists("Bump_new_problem_idun_new_reference/" + filename):
                problem_dictionary = np.load("Bump_new_problem_idun_new_reference/" + filename,
                                             allow_pickle=True).item()
                print("j =", j, "i = ", i, "Lambda = ", lamda)
                print("--------------------------")
                print(problem_dictionary)
                print("--------------------------")
