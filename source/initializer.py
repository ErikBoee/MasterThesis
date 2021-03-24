import numpy as np
import functions as func
from constants import N_TIME, PIXELS, EXACT_RADON_TRANSFORM, BETA
import test_utilities as ut
from skimage.transform import radon

beta = BETA
radius = PIXELS / 3
init_point = np.array([PIXELS / 2, PIXELS / 2 - radius])
init_length = 2 * np.pi * radius
point_ref = np.array([PIXELS / 2, PIXELS / 2 - radius / 1.2])
length_ref = 2 * np.pi * radius
point_sol = point_ref
length_sol = length_ref
angles = [0.93, 1.34, 1.78, 2.08]

t_n = np.linspace(0, 1, N_TIME + 1)


def theta(tau):
    return 2 * np.pi * tau + np.sin(16 * np.pi * tau)


def calc_theta_ref(tau):
    return 2 * np.pi * tau


def calc_gamma_ref():
    return func.calculate_entire_gamma_from_theta(calc_theta_ref(t_n), point_ref, length_ref)


def der_theta(tau):
    return 2 * np.pi + 16 * np.pi * np.cos(16 * np.pi * tau)


def der_theta_ref(tau):
    return 2 * np.pi


theta_ref = calc_theta_ref(t_n)
init_theta = theta_ref
gamma_ref = calc_gamma_ref()


def length_squared(point_1, point_2):
    return (point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2


def geodesic_length_squared(length, gamma, i, j):
    index_difference = min(abs(i - j), (min(i, j) + len(gamma) - 1 - max(i, j)))
    return (index_difference * length / N_TIME) ** 2


def mobius_energy(length, gamma):
    integrand = 0
    for i in range(len(gamma)):
        for j in range(len(gamma)):
            if i != j and abs((i - j)) != len(gamma) - 1:
                integrand += 1 / length_squared(gamma[i], gamma[j]) - 1 / geodesic_length_squared(length, gamma, i, j)
    return integrand / N_TIME**2 * length ** 2


mobius = mobius_energy(length_ref, gamma_ref)
print(mobius)
theta_solution = theta(t_n)
gamma_solution = func.calculate_entire_gamma_from_theta(theta_solution, point_sol, length_sol)
angle_to_exact_radon = {}
filled_radon_image = ut.create_image_from_curve(gamma_solution, PIXELS, t_n)
for angle in angles:
    radon_transform_py = radon(filled_radon_image, theta=[ut.rad_to_deg(angle)], circle=True)
    angle_to_exact_radon[angle] = {EXACT_RADON_TRANSFORM: radon_transform_py}
