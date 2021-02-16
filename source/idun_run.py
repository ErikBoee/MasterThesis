import numpy as np
import utilities_running as ur
import constants as const


max_iterator = 2
filename = "Problems/Circle_test_well_approximation.npy"
ending = "bfgs_tol_lowered"
problem_dictionary = np.load(filename, allow_pickle=True).item()
problem_dictionary["Epsilon for derivative"] = const.EPSILON
problem_dictionary["Delta for heaviside"] = const.DELTA
problem_dictionary["N time"] = const.N_TIME
problem_dictionary["Pixels"] = const.PIXELS
problem_dictionary["Step size"]: const.STEPSIZE
problem_dictionary["Tolerance"] = const.TOL
problem_dictionary["Tolerance penalty"] = const.PENALTY_TOL
problem_dictionary["Max lambda"] = const.MAX_LAMDA
problem_dictionary["Lambda"] = const.LAMDA
opt_object = ur.get_opt_object_from_problem_dictionary_bfgs(problem_dictionary, max_iterator)
ur.test_bfgs_method(problem_dictionary, opt_object, filename, ending)



"""
max_iterator = 5
filename = "Problems/Example_bump_not_created.npy"
problem_dictionary = np.load(filename, allow_pickle=True).item()
problem_dictionary["Epsilon for derivative"] = const.EPSILON
problem_dictionary["Delta for heaviside"] = const.DELTA
problem_dictionary["N time"] = const.N_TIME
problem_dictionary["Pixels"] = const.PIXELS
problem_dictionary["Step size"]: const.STEPSIZE
problem_dictionary["Tolerance"] = const.TOL
problem_dictionary["Tolerance penalty"] = const.PENALTY_TOL
problem_dictionary["Max lambda"] = const.MAX_LAMDA
problem_dictionary["Lambda"] = 0.1
opt_object = ur.get_opt_object_from_problem_dictionary_gd(problem_dictionary, max_iterator)
ur.update_problem_dictionary_and_save(problem_dictionary, opt_object, filename)
"""