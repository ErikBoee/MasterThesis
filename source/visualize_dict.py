import numpy as np
import os

for j in range(10):
    for lamda in [100, 1000, 10000, 100000]:
        for i in range(4):
            filename = "j_" + str(j) + "_i_" + str(i) + "_lambda_" + str(lamda) + ".npy"
            if os.path.exists("Circle_not_fine_grid_update_reference/" + filename):
                problem_dictionary = np.load("Circle_not_fine_grid_update_reference/" + filename,
                                             allow_pickle=True).item()
                print("j =", j, "i = ", i, "Lambda = ", lamda)
                print("--------------------------")
                print(problem_dictionary)
                print("--------------------------")
