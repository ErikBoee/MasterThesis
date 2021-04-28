import numpy as np
import solve_problem_utilities as sp_ut
import parameters as pm
import os
import sys

input_file = sys.argv[1]

problem_dictionary_filename = "Experiments/Experiment_1/" + input_file
new_folder_name = input_file
new_path = '../source/Runs_finished/' + new_folder_name.split('.')[0]

if not os.path.exists(new_path):
    os.makedirs(new_path)
    print("Created path", new_path)
else:
    os.makedirs(new_path + "_strange")
    print("Strange")

directory = new_path
max_iterator = pm.MAX_ITER_BFGS
image_frequency = pm.IMAGE_FREQUENCY
number_of_full_loops = pm.NUMBER_OF_FULL_LOOPS
problem_dictionary = np.load(problem_dictionary_filename, allow_pickle=True).item()
opt_object = sp_ut.get_opt_object_from_problem_dictionary_bfgs(problem_dictionary, max_iterator, image_frequency, number_of_full_loops)
sp_ut.run_bfgs_method(problem_dictionary, opt_object, new_folder_name, new_path)
