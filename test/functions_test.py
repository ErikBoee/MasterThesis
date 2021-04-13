import numpy as np
import cv2
from numba import njit
from constants import DELTA, N_TIME, PIXELS


@njit
def trapezoidal_rule(x, b, a):
    n = len(x)
    interval_lengths = (b - a) / (n - 1)
    return (x[0] + np.sum(2 * x[1:-1]) + x[n - 1]) / 2 * interval_lengths


@njit
def theta(tau):
    return 2 * np.pi * tau + np.sin(16 * np.pi * tau)


@njit
def theta_ref(tau):
    return 2 * np.pi * tau


@njit
def der_theta(tau):
    return 2 * np.pi + 16 * np.pi * np.cos(16 * np.pi * tau)


@njit
def der_theta_ref(tau):
    return 2 * np.pi


@njit
def cos_theta(x):
    return np.cos(theta(x))


@njit
def sin_theta(x):
    return np.sin(theta(x))


@njit
def gamma_i(t_n, i, point, length):
    n = len(t_n) - 1
    assert (i <= n)
    trapezoidal_x_list = np.sum(cos_theta(t_n[1:i])) / n + (cos_theta(t_n[0]) + cos_theta(t_n[i])) / (2 * n)
    trapezoidal_y_list = np.sum(sin_theta(t_n[1:i])) / n + (sin_theta(t_n[0]) + sin_theta(t_n[i])) / (2 * n)
    integral_values = np.array([trapezoidal_x_list, trapezoidal_y_list])
    return np.add(point, np.multiply(length, integral_values))


def der_gamma(t, length):
    return np.multiply(length, [cos_theta(t), sin_theta(t)])


def der_gamma_from_theta(theta, length):
    return np.multiply(length, [np.cos(theta), np.sin(theta)])


def der_gamma_diff_theta(theta, length):
    return_vector = np.multiply(length, [-np.sin(theta[:-1]), np.cos(theta[:-1])])
    return return_vector.T


def gamma_diff_theta(theta, length):
    derivatives = np.zeros((N_TIME + 1, N_TIME, 2))
    for i in range(1, N_TIME + 1):
        for j in range(1, i + 1):
            if j < N_TIME:
                derivatives[i, j - 1] += np.multiply(length / (2 * N_TIME),
                                                     np.array([-np.sin(theta[j - 1]), np.cos(theta[j - 1])]))
                derivatives[i, j] += np.multiply(length / (2 * N_TIME), np.array([-np.sin(theta[j]), np.cos(theta[j])]))
            else:
                derivatives[i, j - 1] += np.multiply(length / (2 * N_TIME),
                                                     np.array([-np.sin(theta[j - 1]), np.cos(theta[j - 1])]))
                derivatives[i, 0] += np.multiply(length / (2 * N_TIME), np.array([-np.sin(theta[j]), np.cos(theta[j])]))

    return derivatives


def calculate_entire_gamma(t_n, point, length):
    entire_gamma = np.zeros((len(t_n), 2))
    entire_gamma[0] = point
    n = len(t_n) - 1
    cos_theta_vector = np.multiply(length / (2 * n), cos_theta(t_n))
    sin_theta_vector = np.multiply(length / (2 * n), sin_theta(t_n))
    cos_sum_vector = cos_theta_vector[:n] + cos_theta_vector[1:]
    sin_sum_vector = sin_theta_vector[:n] + sin_theta_vector[1:]
    cos_cum_sum_vector = np.cumsum(cos_sum_vector)
    sin_cum_sum_vector = np.cumsum(sin_sum_vector)
    cum_sum_vector = np.array([cos_cum_sum_vector, sin_cum_sum_vector]).T
    entire_gamma[1:] = np.add(point, cum_sum_vector)
    return entire_gamma

def calculate_entire_gamma_from_theta(theta, point, length):
    entire_gamma = np.zeros((len(theta), 2))
    entire_gamma[0] = point
    n = len(theta)-1
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


def calculate_entire_gamma_der(t_n, length):
    return der_gamma(t_n, length).T


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
        if i == 0 or i == number_of_points-1:
            integral += integrand(gamma_vector[i], gamma_der_vector[i], basis_vector, alpha)/2
        else:
            integral += integrand(gamma_vector[i], gamma_der_vector[i], basis_vector, alpha)

    return integral/(number_of_points - 1)


@njit
def integrand(gamma_value, gamma_der_value, basis_vector, alpha):
    basis_vector_orthogonal = [-basis_vector[1], basis_vector[0]]
    dot_product = gamma_value[0]*basis_vector_orthogonal[0] + gamma_value[1]*basis_vector_orthogonal[1]
    heaviside = heaviside_cont_num(dot_product - alpha, DELTA)
    second_dot_product_negative = -gamma_der_value[0]*basis_vector[0] - gamma_der_value[1]*basis_vector[1]
    return heaviside * second_dot_product_negative


@njit
def heaviside_cont_num(x, delta):
    return 1 / (1 + np.exp(- 2 * x / delta))


@njit
def heaviside_cont_analytic(x, delta):
    if x <= -delta / 2.0:
        return 0.0
    elif x < delta / 2.0:
        return x / delta + 1 / 2.0
    return 1.0


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
