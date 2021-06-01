import numpy as np
import optimization_object_bfgs_utilities as opt_ut
import matplotlib.pyplot as plt
import constants as const
from visualize_radon_transforms import create_image_from_curve, set_font, radon, rad_to_deg
import os


def get_gammas(problem_dictionary):
    initial_gamma = opt_ut.calculate_entire_gamma_from_theta(problem_dictionary[const.THETA_INITIAL],
                                                             problem_dictionary[const.POINT_INITIAL],
                                                             problem_dictionary[const.LENGTH_INITIAL])
    solution_gamma = opt_ut.calculate_entire_gamma_from_theta(problem_dictionary[const.THETA_SOLUTION],
                                                              problem_dictionary[const.POINT_SOLUTION],
                                                              problem_dictionary[const.LENGTH_SOLUTION])
    reconstructed_gamma = opt_ut.calculate_entire_gamma_from_theta(problem_dictionary[const.THETA_RECONSTRUCTED],
                                                                   problem_dictionary[const.POINT_RECONSTRUCTED],
                                                                   problem_dictionary[const.LENGTH_RECONSTRUCTED])

    initial_gamma[-1] = initial_gamma[0]
    reconstructed_gamma[-1] = reconstructed_gamma[0]
    solution_gamma[-1] = solution_gamma[0]
    return initial_gamma, solution_gamma, reconstructed_gamma


def plot_gammas(initial_gamma, solution_gamma, reconstructed_gamma):
    plt.figure(figsize=[10, 10])
    plt.plot(initial_gamma[:, 0], initial_gamma[:, 1], label="Initial guess", color="dodgerblue")
    plt.plot(reconstructed_gamma[:, 0], reconstructed_gamma[:, 1], color='red', label="Reconstructed")
    plt.plot(solution_gamma[:, 0], solution_gamma[:, 1], label="Solution", color='navy')
    set_axis()
    plt.show()


def visualize_problem(problem_dictionary):
    initial_gamma, solution_gamma, reconstructed_gamma = get_gammas(problem_dictionary)
    plot_gammas(initial_gamma, solution_gamma, reconstructed_gamma)


def set_axis(beta, ax):
    plt.xlabel(r"$x$", fontsize=22)
    plt.ylabel(r"$y$", fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.title(r"$\beta$ = " + beta, fontsize=22)
    leg = plt.legend(frameon=False, fontsize=22)
    plt.draw()  # Draw the figure so you can find the positon of the legend.

    # Get the bounding box of the original legend
    bb = leg.get_bbox_to_anchor().inverse_transformed(ax.transAxes)

    # Change to location of the legend.
    xOffset = 0.12
    bb.x0 += xOffset
    bb.x1 += xOffset
    leg.set_bbox_to_anchor(bb, transform=ax.transAxes)


def visualize_angles(problem_dictionary):
    initial_gamma = opt_ut.calculate_entire_gamma_from_theta(problem_dictionary[const.THETA_INITIAL],
                                                             problem_dictionary[const.POINT_INITIAL],
                                                             problem_dictionary[const.LENGTH_INITIAL])
    initial_gamma[-1] = initial_gamma[0]
    x = np.linspace(0, 200, 201)
    angle_0 = np.ones(201) * 100
    angle_45 = x
    angle_90 = x
    angle_135 = 200 - x
    plt.figure(figsize=[10, 10])
    plt.xlim([0, 200])
    plt.ylim([0, 200])
    plt.plot(initial_gamma[:, 0], initial_gamma[:, 1] - 50, color="red")
    plt.plot(x, angle_0, "--", label="0 degrees", color="dodgerblue")
    plt.plot(x, angle_45, "--", label="45 degrees", color="royalblue")
    plt.plot(angle_0, angle_90, "--", label="90 degrees", color="navy")
    plt.plot(x[55:], angle_135[55:], "--", label="135 degrees", color="blue")
    # set_axis()
    plt.show()


def plot_one_angle(angle, radon_transform_reconstructed, radon_transform_py, pixels):
    set_font()
    plt.figure(figsize=[11, 5])
    plt.title("Radon transform, angle = " + str(round(rad_to_deg(angle))), fontsize=28)
    plt.plot(np.linspace(0, pixels, pixels), radon_transform_reconstructed, color='red',
             label="Reconstructed")
    plt.plot(np.linspace(0, pixels, pixels), radon_transform_py, label="Solution", color='navy')
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(frameon=False, fontsize=28)  # , loc = 'upper')
    # plt.savefig("Figures/Radon_transforms/Circle_test_well_approximation_bfgs" + str(round(rad_to_deg(
    # angle))) + ".pdf")
    plt.show()


def visualize_all_regularizations(experiment_number, no_angles):
    plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
    # Options
    params = {'text.usetex': True,
              'font.size': 22,
              'font.family': 'lmodern'
              }
    plt.rcParams.update(params)
    reg_to_pos = {0.00: 1, 0.01: 2, 0.10: 3, 1.00: 4}
    set_font()
    fig = plt.figure(figsize=[10, 10])
    for filename in os.listdir("Experiments_finished/Experiment_" + str(experiment_number)):
        if str(no_angles) in filename[13:] and not "noise_0_15" in filename:
            filepath = "Experiments_finished/Experiment_" + str(
                experiment_number) + "/" + filename + "/" + filename + ".npy"
            problem_dictionary = np.load(filepath, allow_pickle=True).item()
            pos = reg_to_pos[round(problem_dictionary[const.BETA_STRING] / no_angles, 2)]
            initial_gamma, solution_gamma, reconstructed_gamma = get_gammas(problem_dictionary)
            ax = fig.add_subplot(2, 2, pos)

            plt.plot(initial_gamma[:, 0], initial_gamma[:, 1], label=r"$\gamma_0$", color="dodgerblue")
            plt.plot(reconstructed_gamma[:, 0], reconstructed_gamma[:, 1], color='red', label=r"$\gamma_r$")
            plt.plot(solution_gamma[:, 0], solution_gamma[:, 1], label=r"$\gamma_s$", color='navy')
            set_axis(str(round(problem_dictionary[const.BETA_STRING], 2)), ax)
    plt.show()
    plt.savefig(fname="figures_master/ex_8_no_angles_8_noise.pdf")


def visualize_radon(problem_dictionary_inner):
    reconstructed_gamma, reconstructed_gamma_der, solution_gamma = get_gammas(problem_dictionary_inner)

    pixels = problem_dictionary_inner[const.PIXELS_STRING]

    filled_image = create_image_from_curve(solution_gamma, pixels, solution_gamma)

    set_font()
    plt.figure(figsize=[10, 10])
    i = 1
    for angle in problem_dictionary_inner[const.ANGLES_STRING]:
        radon_transform_py = radon(filled_image, theta=[rad_to_deg(angle)], circle=True)
        plt.subplot(2, 2, i)
        plt.title(str(round(rad_to_deg(angle))) + " degrees", fontsize=30)
        plt.plot(np.linspace(0, pixels, pixels)[49:249], radon_transform_py[49:249], color='red')
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        i += 1
    plt.show()


filename = "Experiments_finished/Experiment_4/Experiment_4_noise_0_beta_4_0_no_angles_4_lambda_100_1000000/Experiment_4_noise_0_beta_4_0_no_angles_4_lambda_100_1000000.npy"
problem_dictionary = np.load(filename, allow_pickle=True).item()
# print(problem_dictionary)
# visualize_problem(problem_dictionary)
# visualize_angles(problem_dictionary)
# visualize_radon(problem_dictionary)
visualize_all_regularizations(8, 5)
