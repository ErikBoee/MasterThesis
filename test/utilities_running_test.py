import numpy as np
import optimization_object_gd as opt
import optimization_object_bfgs as opt_bfgs
import functions as func
from skimage.transform import radon
from constants import EXACT_RADON_TRANSFORM, N_TIME
import constants as const
import numba
from numba import njit


def rad_to_deg(rad):
    return 180 * rad / np.pi


@njit(fastmath=True)
def calculate_winding_number(point_to_wind, entire_gamma):
    gamma_current_subtracted = np.subtract(entire_gamma[:-1], point_to_wind)
    gamma_next_subtracted = np.subtract(entire_gamma[1:], point_to_wind)
    gamma_next_angle = np.arctan2(gamma_next_subtracted[:, 1], gamma_next_subtracted[:, 0])
    gamma_current_angle = np.arctan2(gamma_current_subtracted[:, 1], gamma_current_subtracted[:, 0])
    signed_angles = np.subtract(gamma_next_angle, gamma_current_angle)
    signed_angles[signed_angles > np.pi] -= 2 * np.pi
    signed_angles[signed_angles < -np.pi] += 2 * np.pi
    winding_number = np.sum(signed_angles)
    return round(winding_number / (2 * np.pi))


@njit
def create_image_from_curve(entire_gamma, pixels, t_list):
    img = np.zeros((pixels, pixels), numba.float_)
    boundary_pixels = set()
    number_of_times = len(t_list)
    for i in range(number_of_times):
        x, y = entire_gamma[i]
        img[round(x), round(y)] = 1.0
        boundary_pixels.add((round(x), round(y)))
    for i in range(pixels):
        for j in range(pixels):
            if not (i, j) in boundary_pixels and calculate_winding_number(np.array([i, j]), entire_gamma) == 1:
                img[i, j] = 1.0
    return img


def problem_solver(problem_dictionary, opt_object):
    problem_dictionary["Initial Objective function"] = opt_object.objective_value(opt_object.theta,
                                                                                  opt_object.length,
                                                                                  opt_object.point)
    theta, length, point, iterator, obj_function = opt_object.solver()
    problem_dictionary["Theta reconstructed"] = theta
    problem_dictionary["Point reconstructed"] = point
    problem_dictionary["Length reconstructed"] = length
    problem_dictionary["Iterator"] = iterator
    problem_dictionary["Final Objective function"] = obj_function
    return problem_dictionary


def update_problem_dictionary_and_save(problem_dictionary, opt_object, filename):
    problem_dictionary = problem_solver(problem_dictionary, opt_object)
    np.save(filename[:-4] + "_after_idun", problem_dictionary, allow_pickle=True)
    return problem_dictionary


def test_bfgs_method(problem_dictionary, opt_object, new_file_name, folder_path):
    problem_dictionary["Initial Objective function"] = opt_object.objective_value(opt_object.theta,
                                                                                  opt_object.length,
                                                                                  opt_object.point)
    theta, length, point, iterator, obj_function = opt_object.solver(folder_path)
    problem_dictionary["Theta reconstructed"] = theta
    problem_dictionary["Point reconstructed"] = point
    problem_dictionary["Length reconstructed"] = length
    problem_dictionary["Iterator"] = iterator
    problem_dictionary["Final Objective function"] = obj_function
    np.save(new_file_name, problem_dictionary, allow_pickle=True)
    return problem_dictionary


def get_opt_object_from_problem_dictionary_bfgs(problem_dictionary, max_iterator, image_frequency):
    angle_to_exact_radon, gamma_solution = get_opt_object_from_problem_dictionary(problem_dictionary)
    opt_object = opt_bfgs.OptimizationObjectBFGS(problem_dictionary["Theta initial"],
                                                 problem_dictionary["Length initial"],
                                                 problem_dictionary["Point initial"],
                                                 problem_dictionary["Theta reference"],
                                                 gamma_solution, angle_to_exact_radon, problem_dictionary["Beta"],
                                                 problem_dictionary["Lambda"], problem_dictionary["C_1"],
                                                 problem_dictionary["C_2"],
                                                 problem_dictionary["Tau"],
                                                 max_iterator, image_frequency)

    return opt_object


def update_problem(problem_dictionary, angles):
    problem_dictionary["Epsilon for derivative"] = const.EPSILON
    problem_dictionary["Delta for heaviside"] = const.DELTA
    problem_dictionary["N time"] = const.N_TIME
    problem_dictionary["Pixels"] = const.PIXELS
    problem_dictionary["Step size"]: const.STEPSIZE
    problem_dictionary["Tolerance"] = const.TOL_CONV
    problem_dictionary["Tolerance penalty"] = const.PENALTY_TOL
    problem_dictionary["Max lambda"] = const.MAX_LAMDA_STRING
    problem_dictionary["Lambda"] = const.LAMDA
    problem_dictionary["Beta"] = const.BETA_STRING
    problem_dictionary["Angles"] = angles
    return problem_dictionary


def get_opt_object_from_problem_dictionary_gd(problem_dictionary, max_iterator):
    angle_to_exact_radon, gamma_solution = get_opt_object_from_problem_dictionary(problem_dictionary)

    opt_object = opt.OptimizationObjectGD(problem_dictionary["Theta initial"],
                                          problem_dictionary["Length initial"],
                                          problem_dictionary["Point initial"],
                                          problem_dictionary["Theta reference"],
                                          gamma_solution, angle_to_exact_radon, problem_dictionary["Beta"],
                                          problem_dictionary["Lambda"], problem_dictionary["C"],
                                          problem_dictionary["Tau"],
                                          max_iterator)

    return opt_object


def get_opt_object_from_problem_dictionary(problem_dictionary):
    gamma_solution = func.calculate_entire_gamma_from_theta(problem_dictionary["Theta solution"],
                                                            problem_dictionary["Point solution"],
                                                            problem_dictionary["Length solution"]
                                                            )
    angle_to_exact_radon = {}
    maximum = 0
    for angle in problem_dictionary["Angles"]:
        filled_radon_image = create_image_from_curve(gamma_solution, problem_dictionary["Pixels"],
                                                     np.linspace(0, 1, N_TIME + 1))
        radon_transform_py = radon(filled_radon_image, theta=[rad_to_deg(angle)], circle=True)
        maximum = max(maximum, max(abs(radon_transform_py)))
        angle_to_exact_radon[angle] = {EXACT_RADON_TRANSFORM: radon_transform_py}

    np.random.seed(const.SEED)
    for angle in angle_to_exact_radon.keys():
        angle_to_exact_radon[angle][EXACT_RADON_TRANSFORM][:, 0] += np.random.normal(0, maximum * const.NOISE_SIZE_STRING, len(
            angle_to_exact_radon[angle][EXACT_RADON_TRANSFORM][:, 0]))

    return angle_to_exact_radon, gamma_solution
