import numpy as np
import functions as func
from test_utilities import create_image_from_curve, rad_to_deg
from constants import PIXELS
from skimage.transform import radon
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


def visualize_radon(problem_dictionary):
    reconstructed_gamma = func.calculate_entire_gamma_from_theta(problem_dictionary["Theta reconstructed"],
                                                                 problem_dictionary["Point reconstructed"],
                                                                 problem_dictionary["Length reconstructed"])
    reconstructed_gamma_der = func.calculate_entire_gamma_der_from_theta(problem_dictionary["Theta reconstructed"],
                                                                         problem_dictionary["Length reconstructed"])
    solution_gamma = func.calculate_entire_gamma_from_theta(problem_dictionary["Theta solution"],
                                                            problem_dictionary["Point solution"],
                                                            problem_dictionary["Length solution"])
    reconstructed_gamma[-1] = reconstructed_gamma[0]
    solution_gamma[-1] = solution_gamma[0]
    pixels = problem_dictionary["Pixels"]

    filled_image = create_image_from_curve(solution_gamma, pixels, solution_gamma)
    for angle in problem_dictionary["Angles"]:
        radon_transform_py = radon(filled_image, theta=[rad_to_deg(angle)], circle=True)
        radon_transform_reconstructed = func.radon_transform(reconstructed_gamma, reconstructed_gamma_der,
                                                             angle, pixels)
        radon_transform_reconstructed = np.maximum(radon_transform_reconstructed,
                                                   np.zeros(len(radon_transform_reconstructed)))
        print(radon_transform_reconstructed)
        plt.figure(figsize=[11, 5])
        plt.title("Radon transform, angle = " + str(round(rad_to_deg(angle))), fontsize=24)
        plt.plot(np.linspace(0, pixels, pixels), radon_transform_reconstructed, color='red',
                 label="Reconstructed")
        plt.plot(np.linspace(0, pixels, pixels), radon_transform_py, label="Solution", color='navy')
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.legend(frameon=False, fontsize=28 ) #, loc = 'upper')
        #plt.savefig("Figures/Radon_transforms/Circle_test_well_approximation_bfgs" + str(round(rad_to_deg(
            #angle))) + ".pdf")
        plt.show()


filename = "Problems/Circle_test_not_approximatingtest_on_computer.npy"
problem_dictionary = np.load(filename, allow_pickle=True).item()
print(problem_dictionary)
visualize_radon(problem_dictionary)
print(problem_dictionary["Angles"] * 180 / np.pi)
