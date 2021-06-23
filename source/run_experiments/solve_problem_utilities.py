import numpy as np
import optimization_object_bfgs as opt_bfgs
import optimization_object_bfgs_utilities as opt_ut
from skimage.transform import radon
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


def run_bfgs_method(problem_dictionary, opt_object, new_file_name, folder_path):
    problem_dictionary[const.INITIAL_OBJ_FUNC] = opt_object.objective_function(opt_object.theta,
                                                                            opt_object.length,
                                                                            opt_object.point)
    theta, length, point, iterator, obj_function = opt_object.solver(folder_path)
    problem_dictionary[const.THETA_RECONSTRUCTED] = theta
    problem_dictionary[const.POINT_RECONSTRUCTED] = point
    problem_dictionary[const.LENGTH_RECONSTRUCTED] = length
    problem_dictionary[const.ITERATOR] = iterator
    problem_dictionary[const.FINAL_OBJ_FUNC] = obj_function
    np.save(new_file_name, problem_dictionary, allow_pickle=True)
    return problem_dictionary


def get_opt_object_from_problem_dictionary_bfgs(problem_dictionary, max_iterator, image_frequency, number_of_full_loops):
    angle_to_exact_radon, gamma_solution = get_exact_radon_and_gamma_from_problem_dictionary(problem_dictionary)
    opt_object = opt_bfgs.OptimizationObjectBFGS(problem_dictionary[const.THETA_INITIAL],
                                                 problem_dictionary[const.LENGTH_INITIAL],
                                                 problem_dictionary[const.POINT_INITIAL],
                                                 problem_dictionary[const.THETA_REFERENCE],
                                                 gamma_solution, angle_to_exact_radon,
                                                 problem_dictionary[const.BETA_STRING],
                                                 problem_dictionary[const.LAMBDA_STRING],
                                                 problem_dictionary[const.C_1_STRING],
                                                 problem_dictionary[const.C_2_STRING],
                                                 problem_dictionary[const.TAU_STRING],
                                                 max_iterator, image_frequency,
                                                 problem_dictionary[const.NOISE_SIZE_STRING],
                                                 problem_dictionary[const.MAX_LAMDA_STRING],
                                                 number_of_full_loops)

    return opt_object


def get_exact_radon_and_gamma_from_problem_dictionary(problem_dictionary):
    gamma_solution = opt_ut.calculate_entire_gamma_from_theta(problem_dictionary[const.THETA_SOLUTION],
                                                            problem_dictionary[const.POINT_SOLUTION],
                                                            problem_dictionary[const.LENGTH_SOLUTION]
                                                            )
    angle_to_exact_radon = {}
    maximum = 0
    for angle in problem_dictionary[const.ANGLES_STRING]:
        filled_radon_image = create_image_from_curve(gamma_solution, problem_dictionary[const.PIXELS_STRING],
                                                     np.linspace(0, 1, const.N_TIME + 1))
        radon_transform_py = radon(filled_radon_image, theta=[rad_to_deg(angle)], circle=True)
        maximum = max(maximum, max(abs(radon_transform_py)))
        angle_to_exact_radon[angle] = {const.EXACT_RADON_TRANSFORM: radon_transform_py}

    np.random.seed(const.SEED)
    for angle in angle_to_exact_radon.keys():
        angle_to_exact_radon[angle][const.EXACT_RADON_TRANSFORM][:, 0] += np.random.normal(0,
                                                    maximum * problem_dictionary[const.NOISE_SIZE_STRING],
                                                    len(angle_to_exact_radon[angle][const.EXACT_RADON_TRANSFORM][:, 0]))

    return angle_to_exact_radon, gamma_solution
