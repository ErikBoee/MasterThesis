import numpy as np
import functions as func
import matplotlib.pyplot as plt
import matplotlib as mpl

font_style = 'Times New Roman'
maudfont = {'fontname': font_style, 'size': 16}

mpl.rc('font', family=font_style)

font = {'family': font_style,
        'size': 16}
font_cursive = {'family': font_style, 'style': 'italic',
                'size': 16}
mpl.rc('font', **font)


def visualize_problem(problem_dictionary):
    initial_gamma = func.calculate_entire_gamma_from_theta(problem_dictionary["Theta initial"],
                                                           problem_dictionary["Point initial"],
                                                           problem_dictionary["Length initial"])
    solution_gamma = func.calculate_entire_gamma_from_theta(problem_dictionary["Theta solution"],
                                                            problem_dictionary["Point solution"],
                                                            problem_dictionary["Length solution"])
    reconstructed_gamma = func.calculate_entire_gamma_from_theta(problem_dictionary["Theta reconstructed"],
                                                                 problem_dictionary["Point reconstructed"],
                                                                 problem_dictionary["Length reconstructed"])
    initial_gamma[-1] = initial_gamma[0]
    reconstructed_gamma[-1] = reconstructed_gamma[0]
    solution_gamma[-1] = solution_gamma[0]
    # reference_gamma = initial_gamma - initial_gamma[0] + solution_gamma[0]

    plt.figure(figsize=[8, 6])
    plt.plot(initial_gamma[:, 0], initial_gamma[:, 1], label="Initial guess", color="dodgerblue")
    plt.scatter(initial_gamma[0, 0], initial_gamma[0, 1], s=70, color="black")
    # plt.plot(reference_gamma[:, 0], reference_gamma[:, 1], '--', label="Reference", color="dodgerblue", )
    plt.plot(reconstructed_gamma[:, 0], reconstructed_gamma[:, 1], color='red', label="Reconstructed")
    plt.plot(solution_gamma[:, 0], solution_gamma[:, 1], label="Solution", color='navy')
    plt.xlabel("x", fontsize=14, font=font_cursive)
    plt.ylabel("y", fontsize=14, font=font_cursive)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(frameon=False, fontsize=14)
    plt.show()


def visualize_angles(problem_dictionary):
    initial_gamma = func.calculate_entire_gamma_from_theta(problem_dictionary["Theta initial"],
                                                           problem_dictionary["Point initial"],
                                                           problem_dictionary["Length initial"])
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
    plt.xlabel("x", fontsize=14, font=font_cursive)
    plt.ylabel("y", fontsize=14, font=font_cursive)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(frameon=False, fontsize=14)
    plt.show()


filename = "Problems/Poor_regularization_after_idun.npy"
problem_dictionary = np.load(filename, allow_pickle=True).item()
print(problem_dictionary)
visualize_problem(problem_dictionary)
# visualize_angles(problem_dictionary)
print(problem_dictionary["Angles"] * 180 / np.pi)
