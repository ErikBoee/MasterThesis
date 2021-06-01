import numpy as np
import optimization_object_bfgs_utilities as opt_ut
import constants as const
import os
from numba import njit
from tabulate import tabulate

filename = "Experiments_finished/Experiment_4/Experiment_4_noise_0_beta_4_0_no_angles_4_lambda_100_1000000/Experiment_4_noise_0_beta_4_0_no_angles_4_lambda_100_1000000.npy"
problem_dictionary = np.load(filename, allow_pickle=True).item()


@njit
def error_between_gammas(gamma_rec, gamma_sol):
    no_of_extra_points = 100
    gamma_rec = opt_ut.interpolate(gamma_rec, no_of_extra_points)
    gamma_sol = opt_ut.interpolate(gamma_sol, no_of_extra_points)
    max_dist = furthest_from_solution_rec(gamma_rec, gamma_sol)
    max_dist = furthest_from_rec_solution(gamma_rec, gamma_sol, max_dist)
    return max_dist


@njit
def furthest_from_solution_rec(gamma_rec, gamma_sol):
    max_dist = 0
    for point_rec in gamma_rec:
        min_dist = np.inf
        for point_sol in gamma_sol:
            min_dist = min(min_dist, euclidean_length(point_rec, point_sol))
        max_dist = max(max_dist, min_dist)
    return max_dist


@njit
def furthest_from_rec_solution(gamma_rec, gamma_sol, max_dist):
    for point_sol in gamma_sol:
        min_dist = np.inf
        for point_rec in gamma_rec:
            min_dist = min(min_dist, euclidean_length(point_rec, point_sol))
        max_dist = max(max_dist, min_dist)
    return max_dist


@njit
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


no_of_angles_to_table_row = {4: 1, 5: 2, 8: 3, 16: 4}
reg_to_table_column = {0.00: 0, 0.01: 1, 0.10: 2, 1.00: 3}

experiment_number = 8
table = [['0.0', 0, 0, 0, 0, 0, 0, 0, 0], ['0.01N', 0, 0, 0, 0, 0, 0, 0, 0], ['0.1N', 0, 0, 0, 0, 0, 0, 0, 0],
         ['N', 0, 0, 0, 0, 0, 0, 0, 0]]
for filename in os.listdir("Experiments_finished/Experiment_" + str(experiment_number)):
    filepath = "Experiments_finished/Experiment_" + str(experiment_number) + "/" + filename + "/" + filename + ".npy"
    problem_dictionary = np.load(filepath, allow_pickle=True).item()
    N = len(problem_dictionary[const.ANGLES_STRING])
    j = no_of_angles_to_table_row[N]
    i = reg_to_table_column[round(problem_dictionary[const.BETA_STRING] / N, 2)]
    if problem_dictionary[const.NOISE_SIZE_STRING] > 0:
        table[i][j+4] = round(find_error(problem_dictionary), 2)
    else:
        table[i][j] = round(find_error(problem_dictionary), 2)

    print("Noise:", problem_dictionary[const.NOISE_SIZE_STRING])

print(tabulate(table, headers=["Regularization", r"$A_4$", "$A_5$", "$A_8", "A_16", "$A_4^n$", "$A_5$", "$A_8", "A_16"], tablefmt='latex'))