import numpy as np
import utilities_running as ur

max_iterator = 10
filename = "Problems/Circle_test_well_approximation.npy"
problem_dictionary = np.load(filename, allow_pickle=True).item()
opt_object = ur.get_opt_object_from_problem_dictionary(problem_dictionary, max_iterator)
ur.update_problem_dictionary_and_save(problem_dictionary, opt_object, filename)
