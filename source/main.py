# from source.initializer import beta, theta_ref, angle_to_exact_radon, theta_solution,\
#    gamma_ref, init_point, init_length, init_theta
import source.constants as const
from source.initializer_svg import beta, theta_ref, angle_to_exact_radon, theta_sol, point_sol, length_sol, \
    init_point, init_length, init_theta, gamma_solution, angles
import source.optimization_object as opt
import numpy as np
from os import path

problem_dictionary = {
    "Theta initial": init_theta,
    "Length initial": init_length,
    "Point initial": init_point,
    "Theta reference": theta_ref,
    "Theta solution": theta_sol,
    "Point solution": point_sol,
    "Length solution": length_sol,
    "Beta": const.BETA,
    "Lambda": const.LAMDA,
    "Epsilon for derivative": const.EPSILON,
    "Delta for heaviside": const.DELTA,
    "N time": const.N_TIME,
    "Pixels": const.PIXELS,
    "C": const.C,
    "Tau": const.TAU,
    "Step size": const.STEPSIZE,
    "Tolerance": const.TOL,
    "Number of loops": const.NUMBER_OF_LOOPS,
    "Angles": angles,
    "Tolerance penalty": const.PENALTY_TOL,
    "Max lambda": const.MAX_LAMDA
}

filename = "Circle_test_3"

if __name__ == '__main__':
    if not path.exists(filename + ".npy"):
        opt_object = opt.QuadraticPenalty(init_theta, init_length, init_point, theta_ref,
                                          gamma_solution, angle_to_exact_radon, beta)
        problem_dictionary["Initial Objective function"] = opt_object.objective_function()
        theta, length, point, iterator, obj_function = opt_object.gradient_descent()
        problem_dictionary["Theta reconstructed"] = theta
        problem_dictionary["Point reconstructed"] = point
        problem_dictionary["Length reconstructed"] = length
        problem_dictionary["Iterator"] = length
        problem_dictionary["Final Objective function"] = obj_function
        np.save(filename, problem_dictionary, allow_pickle=True)
    else:
        print("Already created this problem")
