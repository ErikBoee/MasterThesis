import numpy as np
import utilities_running as ur
import constants as const
import os

problem_dictionary_filename = "Problems/Circle_not_fine_grid_update_reference.npy"
new_folder_name = "Circle_not_test_mobius"# + str(const.NOISE_SIZE)
new_path = r'../source/' + new_folder_name
if not os.path.exists(new_path):
    os.makedirs(new_path)
directory = new_path
max_iterator = 10
image_frequency = 1
problem_dictionary = np.load(problem_dictionary_filename, allow_pickle=True).item()
opt_object = ur.get_opt_object_from_problem_dictionary_bfgs(problem_dictionary, max_iterator, image_frequency)
ur.test_bfgs_method(problem_dictionary, opt_object, new_folder_name, new_path)
