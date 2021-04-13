import numpy as np
import utilities_running as ur
import constants as const
import os
name = "Bump_new_problem_not_update_ref"
problem_dictionary_filename = name + "/" + name + ".npy"
new_folder_name = name + "_new_reference"
new_path = r'../source/' + new_folder_name
if not os.path.exists(new_path):
    os.makedirs(new_path)
directory = new_path
max_iterator = 10
image_frequency = 1
problem_dictionary = np.load(problem_dictionary_filename, allow_pickle=True).item()
problem_dictionary["Theta initial"] = problem_dictionary["Theta reconstructed"]
problem_dictionary["Length initial"] = problem_dictionary["Length reconstructed"]
problem_dictionary["Point initial"] = problem_dictionary["Point reconstructed"]
problem_dictionary["Theta reference"] = problem_dictionary["Theta reconstructed"]
opt_object = ur.get_opt_object_from_problem_dictionary_bfgs(problem_dictionary, max_iterator, image_frequency)
ur.run_bfgs_method(problem_dictionary, opt_object, new_folder_name, new_path)