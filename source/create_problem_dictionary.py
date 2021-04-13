import source.constants as const
import parameters as pm
import numpy as np
from source.initializer_svg import theta_ref, theta_sol, point_sol, length_sol, \
    init_point, init_length, init_theta, problem_name

from os import path

problem_dictionary = {
    const.THETA_INITIAL: init_theta,
    const.LENGTH_INITIAL: init_length,
    const.POINT_INITIAL: init_point,
    const.THETA_REFERENCE: theta_ref,
    const.THETA_SOLUTION: theta_sol,
    const.POINT_SOLUTION: point_sol,
    const.LENGTH_SOLUTION: length_sol,
    const.BETA_STRING: pm.BETA,
    const.LAMBDA_STRING: pm.LAMDA,
    const.EPSILON_DERIVATIVE_STRING: const.EPSILON,
    const.DELTA_HEAVISIDE_STRING: const.DELTA,
    const.N_TIME_STRING: const.N_TIME,
    const.PIXELS_STRING: const.PIXELS,
    const.C_1_STRING: pm.C_1,
    const.C_2_STRING: pm.C_2,
    const.TAU_STRING: pm.TAU,
    const.STEPSIZE_STRING: const.STEPSIZE,
    const.TOL_CONV_STRING: const.TOL_CONV,
    const.ANGLES_STRING: pm.ANGLES,
    const.PENALTY_TOL_STRING: const.PENALTY_TOL,
    const.MAX_LAMDA_STRING: pm.MAX_LAMDA,
    const.NOISE_SIZE_STRING: pm.NOISE_SIZE
}

filename = "New_problems/" + problem_name + "_noise_" + str(pm.NOISE_SIZE) + "_beta_" + str(pm.BETA) + "_no_angles_" + str(
    pm.NO_ANGLES) + "_lambda_" + str(pm.LAMDA) + "_" + str(pm.MAX_LAMDA)

filename = filename.replace('.', '_')

if not path.exists(filename):
    np.save(filename, problem_dictionary, allow_pickle=True)
