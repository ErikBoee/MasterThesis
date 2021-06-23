import numpy as np
import optimization_object_bfgs_utilities as opt_ut
import constants as const
import matplotlib.pyplot as plt
import matplotlib as mpl
from test_utilities import create_image_from_curve, rad_to_deg
from skimage.transform import radon


def set_font():
    font_style = 'Times New Roman'

    font = {'family': font_style,
            'size': 16}
    mpl.rc('font', **font)


def get_gammas(problem_dictionary_inner):
    reconstructed_gamma = opt_ut.calculate_entire_gamma_from_theta(problem_dictionary_inner[const.THETA_RECONSTRUCTED],
                                                                   problem_dictionary_inner[const.POINT_RECONSTRUCTED],
                                                                   problem_dictionary_inner[const.LENGTH_RECONSTRUCTED])
    reconstructed_gamma_der = opt_ut.calculate_entire_gamma_der_from_theta(
        problem_dictionary_inner[const.THETA_RECONSTRUCTED],
        problem_dictionary_inner[const.LENGTH_RECONSTRUCTED])

    solution_gamma = opt_ut.calculate_entire_gamma_from_theta(problem_dictionary_inner[const.THETA_SOLUTION],
                                                              problem_dictionary_inner[const.POINT_SOLUTION],
                                                              problem_dictionary_inner[const.LENGTH_SOLUTION])

    reconstructed_gamma[-1] = reconstructed_gamma[0]
    solution_gamma[-1] = solution_gamma[0]

    return reconstructed_gamma, reconstructed_gamma_der, solution_gamma

def plot_one_angle(angle, radon_transform_reconstructed, radon_transform_py, pixels):
    set_font()
    plt.figure(figsize=[11, 5])
    plt.title("Radon transform, angle = " + str(round(rad_to_deg(angle))), fontsize=24)
    plt.plot(np.linspace(0, pixels, pixels), radon_transform_reconstructed, color='red',
             label="Reconstructed")
    plt.plot(np.linspace(0, pixels, pixels), radon_transform_py, label="Solution", color='navy')
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(frameon=False, fontsize=28)  # , loc = 'upper')
    # plt.savefig("Figures/Radon_transforms/Circle_test_well_approximation_bfgs" + str(round(rad_to_deg(
    # angle))) + ".pdf")
    plt.show()


def visualize_radon(problem_dictionary_inner):
    reconstructed_gamma, reconstructed_gamma_der, solution_gamma = get_gammas(problem_dictionary_inner)

    pixels = problem_dictionary_inner[const.PIXELS_STRING]

    filled_image = create_image_from_curve(solution_gamma, pixels, solution_gamma)
    for angle in problem_dictionary_inner[const.ANGLES_STRING]:
        radon_transform_py = radon(filled_image, theta=[rad_to_deg(angle)], circle=True)
        radon_transform_reconstructed = opt_ut.radon_transform(reconstructed_gamma, reconstructed_gamma_der,
                                                               angle, pixels)
        radon_transform_reconstructed = np.maximum(radon_transform_reconstructed,
                                                   np.zeros(len(radon_transform_reconstructed)))
        plot_one_angle(angle, radon_transform_reconstructed,  radon_transform_py, pixels)


filename = "Experiments_finished/Experiment_8/Experiment_8_noise_0_beta_0_5_no_angles_5_lambda_100_1000000/Experiment_8_noise_0_beta_0_5_no_angles_5_lambda_100_1000000.npy"
problem_dictionary = np.load(filename, allow_pickle=True).item()
#visualize_radon(problem_dictionary)
