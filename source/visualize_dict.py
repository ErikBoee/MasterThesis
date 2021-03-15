import numpy as np
import os

for j in range(5):
    for lamda in [100, 1000, 10000, 100000]:
        for i in np.linspace(10, 100, 10):
            filename = "j_" + str(j) + "_lambda_" + str(lamda) + "_i_" + str(int(i)) + ".npy"
            if os.path.exists("Circle_not_fine_grid_update_reference_idun_3/" + filename):
                problem_dictionary = np.load("Circle_not_fine_grid_update_reference_idun_3/" + filename,
                                             allow_pickle=True).item()
                print("j =", j, "i = ", i, "Lambda = ", lamda)
                print("--------------------------")
                print(problem_dictionary)
                print("--------------------------")
