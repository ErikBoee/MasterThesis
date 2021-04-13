import numpy as np
import solve_problem_utilities as sp_ut
import parameters as pm
import os

problem_dictionary_filename = "New_problems/Circle_poor_initial_prob_6_noise_0_0_beta_0_0_no_angles_4_lambda_100_1000000.npy"
new_folder_name = "Circle_poor_initial_prob_6_noise_0_0_beta_0_0_no_angles_4_lambda_100_1000000"
new_path = r'../source/Runs_finished/' + new_folder_name

if not os.path.exists(new_path):
    os.makedirs(new_path)

directory = new_path
max_iterator = pm.MAX_ITER_BFGS
image_frequency = pm.IMAGE_FREQUENCY
number_of_full_loops = pm.NUMBER_OF_FULL_LOOPS
problem_dictionary = np.load(problem_dictionary_filename, allow_pickle=True).item()
opt_object = sp_ut.get_opt_object_from_problem_dictionary_bfgs(problem_dictionary, max_iterator, image_frequency, number_of_full_loops)
sp_ut.run_bfgs_method(problem_dictionary, opt_object, new_folder_name, new_path)
