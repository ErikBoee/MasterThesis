# from source.initializer import beta, theta_ref, angle_to_exact_radon, theta_solution,\
#    gamma_ref, init_point, init_length, init_theta
import source.constants as const
from source.initializer_svg import beta, theta_ref, angle_to_exact_radon, theta_sol, point_sol, length_sol, \
    init_point, init_length, init_theta, gamma_solution, angles, lamda
import source.optimization_object as opt
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
    "Angles": angles,
    "Tolerance penalty": const.PENALTY_TOL,
    "Max lambda": const.MAX_LAMDA
}

filename = "Poor regularization"
max_iterator = 100
if __name__ == '__main__':
    if not path.exists(filename + ".npy"):
        opt_object = opt.QuadraticPenalty(init_theta, init_length, init_point, theta_ref,
                                          gamma_solution, angle_to_exact_radon, beta, lamda, const.C, const.TAU,
                                          max_iterator)
        ur.update_problem_dictionary_and_save(problem_dictionary, opt_object, filename)

    else:
        print("Already created this problem")
