import numpy as np
import optimization_object_bfgs_utilities as opt_ut
import constants as const
import os

filename = "Experiments_finished/Experiment_4/Experiment_4_noise_0_beta_4_0_no_angles_4_lambda_100_1000000/Experiment_4_noise_0_beta_4_0_no_angles_4_lambda_100_1000000.npy"
problem_dictionary = np.load(filename, allow_pickle=True).item()


def error_between_gammas(gamma_rec, gamma_sol):
    no_of_extra_points = 10
    gamma = opt_ut.interpolate(gamma_rec, no_of_extra_points)
    gamma_sol = opt_ut.interpolate(gamma_sol, no_of_extra_points)
    max_dist = 0
    for point_rec in gamma_rec:
        min_dist = np.inf
        for point_sol in gamma_sol:
            min_dist = min(min_dist, euclidean_length(point_rec, point_sol))
        max_dist = max(max_dist, min_dist)
    return max_dist


def euclidean_length(point_1, point_2):
    return np.sqrt((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2)


def find_error(problem_dictionary):
    gamma_rec = opt_ut.calculate_entire_gamma_from_theta(problem_dictionary[const.THETA_RECONSTRUCTED],
                                                         problem_dictionary[const.POINT_RECONSTRUCTED],
                                                         problem_dictionary[const.LENGTH_RECONSTRUCTED])
    gamma_sol = opt_ut.calculate_entire_gamma_from_theta(problem_dictionary[const.THETA_SOLUTION],
                                                         problem_dictionary[const.POINT_SOLUTION],
                                                         problem_dictionary[const.LENGTH_SOLUTION])
    return error_between_gammas(gamma_rec, gamma_sol)

experiment_number = 6
for filename in os.listdir("Experiments_finished/Experiment_" + str(experiment_number)):
    filepath = "Experiments_finished/Experiment_" + str(experiment_number) + "/" + filename + "/" + filename + ".npy"
    print("----------------------------------------")
    problem_dictionary = np.load(filepath, allow_pickle=True).item()
    print("Noise:", problem_dictionary[const.NOISE_SIZE_STRING])
    print("Beta:", problem_dictionary[const.BETA_STRING])
    print("Number of angles:", len(problem_dictionary[const.ANGLES_STRING]))
    print("Error:", find_error(problem_dictionary))
    print("----------------------------------------")
