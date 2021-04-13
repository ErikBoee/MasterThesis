import numpy as np
import cv2
from numba import njit
from constants import DELTA


@njit
def length_squared(point_1, point_2):
    return (point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2


@njit
def geodesic_length_squared(gamma, length, i, j, n):
    index_difference = min(abs(i - j), (min(i, j) + len(gamma) - 1 - max(i, j)))
    return (index_difference * length / n) ** 2

@njit
def mobius_energy(gamma, length, n):
    integrand = 0
    for i in range(len(gamma)):
        for j in range(len(gamma)):
            if i != j and abs((i - j)) != len(gamma) - 1:
                integrand += 1 / length_squared(gamma[i], gamma[j]) - 1 / geodesic_length_squared(gamma, length, i, j,
                                                                                                  n)
    return integrand / n ** 2 * length ** 2


@njit
def trapezoidal_rule(x, b, a):
    n = len(x)
    interval_lengths = (b - a) / (n - 1)
    return (x[0] + np.sum(2 * x[1:-1]) + x[n - 1]) / 2 * interval_lengths


def der_gamma_from_theta(theta, length):
    return np.multiply(length, [np.cos(theta), np.sin(theta)])


def calculate_entire_gamma_from_theta(theta, point, length):
    entire_gamma = np.zeros((len(theta), 2))
    entire_gamma[0] = point
    n = len(theta) - 1
    cos_theta_vector = np.multiply(length / (2 * n), np.cos(theta))
    sin_theta_vector = np.multiply(length / (2 * n), np.sin(theta))
    cos_sum_vector = cos_theta_vector[:n] + cos_theta_vector[1:]
    sin_sum_vector = sin_theta_vector[:n] + sin_theta_vector[1:]
    cos_cum_sum_vector = np.cumsum(cos_sum_vector)
    sin_cum_sum_vector = np.cumsum(sin_sum_vector)
    cum_sum_vector = np.array([cos_cum_sum_vector, sin_cum_sum_vector]).T
    entire_gamma[1:] = np.add(point, cum_sum_vector)
    return entire_gamma


def calculate_entire_gamma_der_from_theta(theta, length):
    return der_gamma_from_theta(theta, length).T


@njit
def energy_function(theta_vector, theta_ref_vector):
    n = len(theta_vector) - 1
    derivatives_theta = np.multiply(n, theta_vector[1:] - theta_vector[0:n])
    derivatives_theta_ref = np.multiply(n, theta_ref_vector[1:] - theta_ref_vector[0:n])
    trapezoidal_integral = sum((derivatives_theta[1:(n - 1)] - derivatives_theta_ref[1:(n - 1)]) ** 2)
    trapezoidal_integral += (derivatives_theta[0] - derivatives_theta_ref[0]) ** 2 / 2
    trapezoidal_integral += (derivatives_theta[n - 1] - derivatives_theta_ref[n - 1]) ** 2 / 2
    return trapezoidal_integral


@njit(fastmath=True)
def get_alphas(angle, pixels):
    assert (0 <= angle <= np.pi)
    length_to_center = pixels / np.sqrt(2)
    if angle <= np.pi / 2:
        skew = 0
        if angle < np.pi / 4:
            skew = np.sin(np.pi / 4 - angle) * length_to_center
        elif np.pi / 4 < angle <= np.pi / 2:
            skew = -np.sin(angle - np.pi / 4) * length_to_center
    else:
        if angle <= 3 * np.pi / 4:
            skew = -np.cos(np.pi / 4 - (angle - np.pi / 2)) * length_to_center
        else:
            skew = -np.cos((angle - np.pi / 2) - np.pi / 4) * length_to_center

    return np.linspace(-round(pixels / 2 - skew), round(pixels / 2 + skew), pixels)


@njit
def radon_transform(gamma_vector, gamma_der_vector, angle, pixels):
    alphas = get_alphas(angle, pixels)
    basis_vector = np.array([np.cos(angle), np.sin(angle)])
    number_of_alphas = len(alphas)
    radons = np.zeros(number_of_alphas)
    for i in range(number_of_alphas):
        radons[i] = integrate_for_radon(gamma_vector, gamma_der_vector, alphas[i], basis_vector)
    return radons


@njit
def integrate_for_radon(gamma_vector, gamma_der_vector, alpha, basis_vector):
    number_of_points = len(gamma_vector)
    integral = 0
    for i in range(number_of_points):
        if i == 0 or i == number_of_points - 1:
            integral += integrand(gamma_vector[i], gamma_der_vector[i], basis_vector, alpha) / 2
        else:
            integral += integrand(gamma_vector[i], gamma_der_vector[i], basis_vector, alpha)

    return integral / (number_of_points - 1)


@njit
def integrand(gamma_value, gamma_der_value, basis_vector, alpha):
    basis_vector_orthogonal = [-basis_vector[1], basis_vector[0]]
    dot_product = gamma_value[0] * basis_vector_orthogonal[0] + gamma_value[1] * basis_vector_orthogonal[1]
    heaviside = heaviside_cont_num(dot_product - alpha, DELTA)
    second_dot_product_negative = -gamma_der_value[0] * basis_vector[0] - gamma_der_value[1] * basis_vector[1]
    return heaviside * second_dot_product_negative


@njit
def heaviside_cont_num(x, delta):
    return 1 / (1 + np.exp(- 2 * x / delta))


def draw_boundary(gamma, gamma_ref, iterator, pixels):
    boundary_image = get_boundary_image(gamma, gamma_ref, pixels)
    cv2.imwrite(str(pixels) + "_x_" + str(pixels) + "_" + str(iterator) + "_boundary.png", boundary_image)


def get_boundary_image(gamma, gamma_ref, pixels):
    boundary_image = np.zeros((pixels, pixels, 3), np.uint8)
    for gamma_value in gamma:
        if -1 < int(gamma_value[0]) < pixels and -1 < int(gamma_value[1]) < pixels:
            boundary_image[round(gamma_value[0]), round(gamma_value[1])] = [255, 255, 255]

    for gamma_value in gamma_ref:
        boundary_image[round(gamma_value[0]), round(gamma_value[1])] = [0, 0, 255]

    return boundary_image
