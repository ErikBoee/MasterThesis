import numpy as np
import source.optimization_object as opt
import source.functions as func
import test_utilities as ut
from skimage.transform import radon
from source.constants import EXACT_RADON_TRANSFORM


def update_problem_dictionary_and_save(problem_dictionary, opt_object, filename):
    problem_dictionary["Initial Objective function"] = opt_object.objective_function()
    theta, length, point, iterator, obj_function = opt_object.gradient_descent()
    problem_dictionary["Theta reconstructed"] = theta
    problem_dictionary["Point reconstructed"] = point
    problem_dictionary["Length reconstructed"] = length
    problem_dictionary["Iterator"] = iterator
    problem_dictionary["Final Objective function"] = obj_function
    np.save(filename, problem_dictionary, allow_pickle=True)
    return problem_dictionary


def get_opt_object_from_problem_dictionary(problem_dictionary):
    gamma_solution = func.calculate_entire_gamma_from_theta(problem_dictionary["Theta solution"],
                                                            problem_dictionary["Length solution"],
                                                            problem_dictionary["Point solution"])
    angle_to_exact_radon = {}
    for angle in problem_dictionary["Angles"]:
        filled_radon_image = ut.create_image_from_curve(gamma_solution, problem_dictionary["Pixels"],
                                                        np.linspace(0, 1, problem_dictionary["N time"] + 1))
        radon_transform_py = radon(filled_radon_image, theta=[ut.rad_to_deg(angle)], circle=True)
        angle_to_exact_radon[angle] = {EXACT_RADON_TRANSFORM: radon_transform_py}

    opt_object = opt.QuadraticPenalty(problem_dictionary["Theta initial"], problem_dictionary["Length initial"],
                                      problem_dictionary["Point initial"], problem_dictionary["Theta reference"],
                                      gamma_solution, angle_to_exact_radon, problem_dictionary["Beta"],
                                      problem_dictionary["Lambda"], problem_dictionary["C"], problem_dictionary["Tau"])

    return opt_object
