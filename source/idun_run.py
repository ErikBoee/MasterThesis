import numpy as np
import source.utilities_running as ur

filename = "Good_example_first.npy"
problem_dictionary = np.load(filename, allow_pickle=True).item()
opt_object = ur.get_opt_object_from_problem_dictionary(problem_dictionary)
ur.update_problem_dictionary_and_save(problem_dictionary, opt_object, filename)
