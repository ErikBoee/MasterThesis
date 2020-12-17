import numpy as np
import utilities_running as ur
import constants as const

max_iterator = 10000
filename = "Problems/Good_example.npy"
problem_dictionary = np.load(filename, allow_pickle=True).item()
problem_dictionary["Epsilon for derivative"] = const.EPSILON
problem_dictionary["Delta for heaviside"] = const.DELTA
problem_dictionary["N time"] = const.N_TIME
problem_dictionary["Pixels"] = const.PIXELS
problem_dictionary["Step size"]: const.STEPSIZE
problem_dictionary["Tolerance"] = const.TOL
problem_dictionary["Tolerance penalty"] = const.PENALTY_TOL
problem_dictionary["Max lambda"] = const.MAX_LAMDA
opt_object = ur.get_opt_object_from_problem_dictionary(problem_dictionary, max_iterator)
ur.update_problem_dictionary_and_save(problem_dictionary, opt_object, filename)
