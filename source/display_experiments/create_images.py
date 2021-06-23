import os
import numpy as np
import optimization_object_bfgs_utilities as opt_ut
import constants as const

experiment_number = 5
for filename in os.listdir("Experiments_finished/Experiment_" + str(experiment_number)):
    path_name = "Experiments_finished/Experiment_" + str(experiment_number) + "/" + filename + "/"
    print(path_name)
    problem_dictionary_filename = "Experiments_finished/Experiment_" + str(experiment_number) + "/" + filename + "/" + filename + ".npy"
    if os.path.exists(problem_dictionary_filename):
        problem_dictionary = np.load(problem_dictionary_filename, allow_pickle=True).item()
        gamma_solution = opt_ut.calculate_entire_gamma_from_theta(problem_dictionary[const.THETA_SOLUTION],
                                                                  problem_dictionary[const.POINT_SOLUTION],
                                                                  problem_dictionary[const.LENGTH_SOLUTION]
                                                                  )
        gamma_reconstructed = opt_ut.calculate_entire_gamma_from_theta(problem_dictionary[const.THETA_RECONSTRUCTED],
                                                                  problem_dictionary[const.POINT_RECONSTRUCTED],
                                                                  problem_dictionary[const.LENGTH_RECONSTRUCTED]
                                                                  )
        opt_ut.draw_boundary_finished(gamma_solution, gamma_reconstructed, const.PIXELS, path_name)