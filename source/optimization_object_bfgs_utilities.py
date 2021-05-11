import numpy as np
import cv2
from numba import njit
from constants import DELTA, PIXELS


@njit
def length_squared(point_1, point_2):
    return ((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2) ** 4


@njit
def path_lengths(no_of_extra_points, i, j, length_vector):
    lowest = min(i, j)
    highest = max(i, j)
    lowest_remainder = lowest % (no_of_extra_points + 1)
    highest_remainder = highest % (no_of_extra_points + 1)
    not_crossing_start_path_length = get_not_crossing_start_path_length(lowest, highest, lowest_remainder,
                                                                        highest_remainder, length_vector,
                                                                        no_of_extra_points)
    crossing_start_path_length = get_crossing_start_path_length(lowest, highest, lowest_remainder, highest_remainder,
                                                                length_vector, no_of_extra_points)

    return crossing_start_path_length, not_crossing_start_path_length


@njit
def get_not_crossing_start_path_length(lowest, highest, lowest_remainder,
                                       highest_remainder, length_vector,
                                       no_of_extra_points):
    index_lowest = int(np.floor(lowest / (no_of_extra_points + 1)))
    index_highest = int(np.floor(highest / (no_of_extra_points + 1)))
    if index_highest == index_lowest:
        return length_vector[index_lowest] * (highest_remainder - lowest_remainder) / (no_of_extra_points + 1)
    length = 0
    for i in range(index_highest - index_lowest + 1):
        if i == 0:
            length += length_vector[index_lowest] * (no_of_extra_points + 1 - lowest_remainder) / (
                    no_of_extra_points + 1)
        elif i == index_highest - index_lowest:
            length += length_vector[index_highest - 1] * highest_remainder / (
                    no_of_extra_points + 1)
        else:
            length += length_vector[index_lowest + i]
    return length


@njit
def get_crossing_start_path_length(lowest, highest, lowest_remainder,
                                   highest_remainder, length_vector,
                                   no_of_extra_points):
    length_path_from_initial = get_not_crossing_start_path_length(0, lowest, 0, lowest_remainder, length_vector,
                                                                  no_of_extra_points)
    length_path_to_initial = get_not_crossing_start_path_length(highest,
                                                                len(length_vector) * (no_of_extra_points + 1) - 1,
                                                                highest_remainder, no_of_extra_points + 1,
                                                                length_vector,
                                                                no_of_extra_points)
    return length_path_from_initial + length_path_to_initial


@njit
def mobius_energy(gamma, length, n, obj_value, length_vector):
    no_of_extra_points = get_number_of_extra_points(n - 1, length, obj_value)
    gamma = interpolate(gamma, no_of_extra_points)
    length_gamma = len(gamma)
    integrand = 0
    crossing_start_path_length_i, not_crossing_start_path_length_i = path_lengths(no_of_extra_points, 0, 0,
                                                                                  length_vector)
    for i in range(length_gamma):
        i_integrand = integrate_distance_for_i(i, crossing_start_path_length_i, not_crossing_start_path_length_i,
                                               no_of_extra_points, length_vector, length_gamma, gamma)

        crossing_start_path_length_i, not_crossing_start_path_length_i = update_path_lengths_i(i,
                no_of_extra_points, length_vector, crossing_start_path_length_i, not_crossing_start_path_length_i)

        integrand += add_i_integrand(i, i_integrand, length_gamma)

    return integrand / (length_gamma - 1) ** 2 * length ** 2

@njit
def integrate_distance_for_i(i, crossing_start_path_length_i, not_crossing_start_path_length_i,
                                               no_of_extra_points, length_vector, length_gamma, gamma):
    i_integrand = 0
    not_crossing_start_path_length_j = not_crossing_start_path_length_i
    crossing_start_path_length_j = crossing_start_path_length_i
    for j in range(length_gamma):
        i_integrand += add_to_i_integrand(gamma, i, j, no_of_extra_points, length_gamma,
                                          not_crossing_start_path_length_j,
                                          crossing_start_path_length_j)
        crossing_start_path_length_j, not_crossing_start_path_length_j = update_path_lengths_j(i, j,
                                                                                               no_of_extra_points,
                                                                                               length_vector,
                                                                                               crossing_start_path_length_j,
                                                                                               not_crossing_start_path_length_j)
    return i_integrand


@njit
def add_i_integrand(i, i_integrand, length_gamma):
    if i == 0 or i == length_gamma - 1:
        return i_integrand / 2
    return i_integrand


@njit
def update_path_lengths_i(i, no_of_extra_points, length_vector, crossing_start_path_length_i,
                          not_crossing_start_path_length_i):
    index_i = int(np.floor(i / (no_of_extra_points + 1)))
    length_element_i = length_vector[index_i] / (no_of_extra_points + 1)
    crossing_start_path_length_i -= length_element_i
    not_crossing_start_path_length_i += length_element_i
    return crossing_start_path_length_i, not_crossing_start_path_length_i


@njit
def update_path_lengths_j(i, j, no_of_extra_points, length_vector, crossing_start_path_length_j,
                          not_crossing_start_path_length_j):
    index_j = int(np.floor(j / (no_of_extra_points + 1)))
    length_element_j = length_vector[index_j] / (no_of_extra_points + 1)
    if j >= i:
        crossing_start_path_length_j -= length_element_j
        not_crossing_start_path_length_j += length_element_j
    else:
        crossing_start_path_length_j += length_element_j
        not_crossing_start_path_length_j -= length_element_j
    return crossing_start_path_length_j, not_crossing_start_path_length_j


@njit
def add_to_i_integrand(gamma, i, j, no_of_extra_points, length_gamma, clockwise_path_length,
                       counter_clockwise_path_length):
    if validation_check(i, j, no_of_extra_points, length_gamma):
        euc_length = length_squared(gamma[i], gamma[j])
        geo_length = min(clockwise_path_length, counter_clockwise_path_length) ** 8
        if abs(euc_length) < 10 ** (-16) or abs(geo_length) < 10 ** (-16):
            return 10 ** 16
        add_to_int = 1 / euc_length - 1 / geo_length
        if j == 0 or j == len(gamma) - 1:
            return max(0, add_to_int / 2)
        else:
            return max(0, add_to_int)
    return 0


@njit
def validation_check(i, j, no_of_extra_points, length_gamma):
    modulo = i % (1 + no_of_extra_points)
    if i < no_of_extra_points + 1:
        return 2 * (1 + no_of_extra_points) < j < length_gamma - 2 - no_of_extra_points

    elif i > length_gamma - 3 - no_of_extra_points:
        return no_of_extra_points + 1 < j < length_gamma - 1 - 2 * (1 + no_of_extra_points)

    else:
        return i + (no_of_extra_points + 1) - modulo + 1 + no_of_extra_points < j or j < i - modulo - (
                1 + no_of_extra_points)


@njit
def get_number_of_extra_points(len_gamma, length, obj_value):
    if obj_value == 1:
        return 10
    else:
        value = int(np.ceil((obj_value ** (1 / 8) * abs(length) + 1) / (np.sqrt(2) * len_gamma) - 1))
        return min(20, value)


@njit
def interpolate(gamma, no_of_extra_points):
    new_gamma = np.zeros((((len(gamma) - 1) * (1 + no_of_extra_points) + 1), 2))
    index_multiplier = (1 + no_of_extra_points)
    for i in range(len(gamma) - 1):
        new_gamma[i * index_multiplier] = gamma[i]
        for j in range(no_of_extra_points):
            new_gamma[i * index_multiplier + j + 1] = (j + 1) / index_multiplier * gamma[i + 1] + (
                    no_of_extra_points - j) / index_multiplier * gamma[i]
    new_gamma[-1] = gamma[-1]
    return new_gamma


@njit
def trapezoidal_rule(x, b, a):
    n = len(x)
    interval_lengths = (b - a) / (n - 1)
    return (x[0] + np.sum(2 * x[1:-1]) + x[n - 1]) / 2 * interval_lengths


def der_gamma_from_theta(theta, length):
    return np.multiply(length, [np.cos(theta), np.sin(theta)])


def get_length_vector(theta, length):
    sin_sum_vector, cos_sum_vector = get_cos_sum_and_sin_sum(theta, length)
    return np.sqrt(sin_sum_vector ** 2 + cos_sum_vector ** 2)


def get_cos_sum_and_sin_sum(theta, length):
    n = len(theta) - 1
    cos_theta_vector = np.multiply(length / (2 * n), np.cos(theta))
    sin_theta_vector = np.multiply(length / (2 * n), np.sin(theta))
    cos_sum_vector = cos_theta_vector[:n] + cos_theta_vector[1:]
    sin_sum_vector = sin_theta_vector[:n] + sin_theta_vector[1:]
    return sin_sum_vector, cos_sum_vector


def calculate_entire_gamma_from_theta(theta, point, length):
    entire_gamma = np.zeros((len(theta), 2))
    entire_gamma[0] = point
    sin_sum_vector, cos_sum_vector = get_cos_sum_and_sin_sum(theta, length)
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

def draw_boundary_finished(gamma_sol, gamma_reconstructed, pixels, path_name):
    boundary_image = get_boundary_image(gamma_reconstructed, gamma_sol, pixels)
    cv2.imwrite(path_name + "Finished_reconstructed_boundary.png", boundary_image)

def get_boundary_image(gamma, gamma_ref, pixels):
    boundary_image = np.zeros((pixels, pixels, 3), np.uint8)
    for gamma_value in gamma:
        if -1 < int(gamma_value[0]) < pixels and -1 < int(gamma_value[1]) < pixels:
            boundary_image[round(gamma_value[0]), round(gamma_value[1])] = [255, 255, 255]

    for gamma_value in gamma_ref:
        boundary_image[round(gamma_value[0]), round(gamma_value[1])] = [0, 0, 255]

    return boundary_image
