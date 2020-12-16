import numpy as np
import functions as func
import matplotlib.pyplot as plt

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

    plt.plot(initial_gamma[:, 0], initial_gamma[:, 1], label="Initial")
    plt.plot(reconstructed_gamma[:, 0], reconstructed_gamma[:, 1], label="reconstructed")
    plt.plot(solution_gamma[:, 0], solution_gamma[:, 1], label="solution")
    plt.legend()
    plt.savefig('testplot.pdf')
    plt.show()


filename = "../Problems/Circle_test_well_approximation.npy"
problem_dictionary = np.load(filename, allow_pickle=True).item()
visualize_problem(problem_dictionary)



