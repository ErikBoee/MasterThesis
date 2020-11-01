import numpy as np
import source.functions as func
from source.constants import N_TIME, PIXELS, EXACT_RADON_TRANSFORM
import test_utilities as ut
from skimage.transform import radon

beta = 0.1
radius = PIXELS / 3
init_point = (PIXELS / 2, PIXELS / 2 - radius)
init_length = 2 * np.pi * radius
point_ref = (PIXELS / 2, PIXELS / 2 - radius)
length_ref = 2 * np.pi * radius
point_sol = point_ref
length_sol = length_ref
angles = np.linspace(0, np.pi, 5)

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
theta_solution = theta(t_n)
gamma_solution = func.calculate_entire_gamma_from_theta(theta_solution, point_sol, length_sol)
angle_to_exact_radon = {}
for angle in angles:
    filled_radon_image = ut.create_image_from_curve(gamma_solution, PIXELS, t_n)
    radon_transform_py = radon(filled_radon_image, theta=[ut.rad_to_deg(angle)], circle=True)
    angle_to_exact_radon[angle] = {EXACT_RADON_TRANSFORM: radon_transform_py}
