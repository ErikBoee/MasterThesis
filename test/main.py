import parameters as pm
import source.constants as const
from initializer_svg_test import theta_ref, angle_to_exact_radon, theta_sol, point_sol, length_sol, \
    init_point, init_length, init_theta, gamma_solution
import source.optimization_object_bfgs as opt_bfgs
from os import path
import source.utilities_running as ur



problem_dictionary = {
    "Theta initial": init_theta,
    "Length initial": init_length,
    "Point initial": init_point,
    "Theta reference": theta_ref,
    "Theta solution": theta_sol,
    "Point solution": point_sol,
    "Length solution": length_sol,
    "Beta": pm.BETA,
    "Lambda": pm.LAMDA,
    "Epsilon for derivative": const.EPSILON,
    "Delta for heaviside": const.DELTA,
    "N time": const.N_TIME,
    "Pixels": const.PIXELS,
    "C": pm.C_1,
    "Tau": pm.TAU,
    "Step size": const.STEPSIZE,
    "Tolerance": const.TOL_CONV,
    "Angles": pm.angles,
    "Tolerance penalty": const.PENALTY_TOL,
    "Max lambda": pm.MAX_LAMDA
}

filename = "Circle_not_fine_grid_update_reference"
max_iterator = 50
image_frequency = 10
name = "Test_iterative_approach"
if __name__ == '__main__':
    if not path.exists(filename + ".npy"):
        opt_object = opt_bfgs.OptimizationObjectBFGS(init_theta, init_length, init_point, theta_ref,
                                                     gamma_solution, angle_to_exact_radon, pm.BETA, pm.LAMDA, pm.C_1, pm.C_2, const.TAU_STRING,
                                                     max_iterator, image_frequency, name)
        ur.update_problem_dictionary_and_save(problem_dictionary, opt_object, filename)

    else:
        print("Already created this problem")
