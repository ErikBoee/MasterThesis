import source.constants as const
import numpy as np
from source.initializer_svg import beta, theta_ref, angle_to_exact_radon, theta_sol, point_sol, length_sol, \
    init_point, init_length, init_theta, gamma_solution, angles, lamda

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
    "C_1": const.C_1,
    "C_2": const.C_2,
    "Tau": const.TAU,
    "Step size": const.STEPSIZE,
    "Tolerance": const.TOL,
    "Angles": angles,
    "Tolerance penalty": const.PENALTY_TOL,
    "Max lambda": const.MAX_LAMDA,
    "Noise size": const.NOISE_SIZE
}

filename = "Problems/Bump_new_problem_noise_" + str(const.NOISE_SIZE)

if not path.exists(filename):
 np.save(filename, problem_dictionary, allow_pickle=True)
