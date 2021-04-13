import numpy as np
import optimization_object_bfgs_utilities as opt_ut
import matplotlib.pyplot as plt
import constants as const


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
    plt.figure(figsize=[8, 6])
    plt.plot(initial_gamma[:, 0], initial_gamma[:, 1], label="Initial guess", color="dodgerblue")
    plt.scatter(initial_gamma[0, 0], initial_gamma[0, 1], s=70, color="black")
    plt.plot(reconstructed_gamma[:, 0], reconstructed_gamma[:, 1], color='red', label="Reconstructed")
    plt.plot(solution_gamma[:, 0], solution_gamma[:, 1], label="Solution", color='navy')
    set_axis()
    plt.show()

def visualize_problem(problem_dictionary):
    initial_gamma, solution_gamma, reconstructed_gamma = get_gammas(problem_dictionary)
    plot_gammas(initial_gamma, solution_gamma, reconstructed_gamma)


def set_axis():
    font_style = 'Times New Roman'
    font_cursive = {'family': font_style, 'style': 'italic',
                    'size': 16}
    plt.xlabel("x", fontsize=14, font=font_cursive)
    plt.ylabel("y", fontsize=14, font=font_cursive)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(frameon=False, fontsize=14)

def visualize_angles(problem_dictionary):
    initial_gamma = opt_ut.calculate_entire_gamma_from_theta(problem_dictionary[const.THETA_INITIAL],
                                                             problem_dictionary[const.POINT_INITIAL],
                                                             problem_dictionary[const.LENGTH_INITIAL])
    initial_gamma[-1] = initial_gamma[0]
    x = np.linspace(0, 200, 201)
    angle_0 = np.ones(201) * 100
    angle_45 = x
    angle_90 = x
    plt.figure(figsize=[8, 6])
    plt.xlim([0, 200])
    plt.ylim([0, 200])
    plt.plot(initial_gamma[:, 0], initial_gamma[:, 1], color="red")
    plt.plot(x, angle_0, "--", label="0 degrees", color="dodgerblue")
    plt.plot(x, angle_45, "--", label="45 degrees", color="royalblue")
    plt.plot(angle_0, angle_90, "--", label="90 degrees", color="navy")
    set_axis()
    plt.show()

filename = "Runs_finished/Star_prob_5_noise_0_0_beta_0_5_no_angles_8_lambda_100_1000/Star_prob_5_noise_0_0_beta_0_5_no_angles_8_lambda_100_1000.npy"
problem_dictionary = np.load(filename, allow_pickle=True).item()
print(problem_dictionary)
visualize_problem(problem_dictionary)
# visualize_angles(problem_dictionary)
