import numpy as np


def theta(tau):
    return 2 * np.pi * tau + np.sin(16 * np.pi * tau)


def theta_ref(tau):
    return 2 * np.pi * tau


def der_theta(tau):
    return 2 * np.pi + 16 * np.pi * np.cos(16 * np.pi * tau)


def der_theta_ref(tau):
    return 2 * np.pi


def cos_theta(x):
    return np.cos(theta(x))


def sin_theta(x):
    return np.sin(theta(x))


def gamma_i(t_n, i, point, length):
    x_integral = 0
    y_integral = 0
    n = len(t_n) - 1
    assert (i <= n)
    for j in range(i):
        x_integral += (cos_theta(t_n[j]) + cos_theta(t_n[j + 1])) / (2 * n)
        y_integral += (sin_theta(t_n[j]) + sin_theta(t_n[j + 1])) / (2 * n)
    integral_values = np.array([x_integral, y_integral])
    return np.add(point, np.multiply(length, integral_values))


def der_gamma_i(t_n, i, length):
    return np.multiply(length, [cos_theta(t_n[i]), sin_theta(t_n[i])])


def calculate_entire_gamma(t_n, point, length):
    entire_gamma = np.zeros((len(t_n), 2))
    entire_gamma[0] = point
    n = len(t_n) - 1
    for j in range(1, n + 1):
        x_value_to_add = length * (cos_theta(t_n[j - 1]) + cos_theta(t_n[j])) / (2 * n)
        y_value_to_add = length * (sin_theta(t_n[j - 1]) + sin_theta(t_n[j])) / (2 * n)
        entire_gamma[j] = np.add(entire_gamma[j - 1], [x_value_to_add, y_value_to_add])
    return entire_gamma


def calculate_entire_gamma_der(t_n, length):
    entire_gamma_der = np.zeros((len(t_n), 2))
    n = len(t_n)
    for j in range(n):
        entire_gamma_der[j] = der_gamma_i(t_n, j, length)
    return entire_gamma_der


def energy_function(t_n):
    integral = 0
    n = len(t_n) - 1
    for i in range(n - 1):
        integral += n / 2 * (((theta(t_n[i + 1]) - theta(t_n[i])) - (theta_ref(t_n[i + 1]) - theta_ref(t_n[i]))) ** 2 +
                             ((theta(t_n[i + 2]) - theta(t_n[i + 1])) - (
                                     theta_ref(t_n[i + 2]) - theta_ref(t_n[i + 1]))) ** 2)
    integral += n * ((theta(t_n[n]) - theta(t_n[n - 1])) - (theta_ref(t_n[n]) - theta_ref(t_n[n - 1]))) ** 2
    return integral


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
            skew = -np.cos(np.pi / 4 - (angle - np.pi/2)) * length_to_center
        else:
            skew = -np.cos((angle - np.pi/2) - np.pi / 4) * length_to_center

    return np.linspace(-round(pixels / 2 - skew), round(pixels / 2 + skew), pixels)


def radon_transform(gamma_vector, gamma_der_vector, angle, pixels):
    alphas = get_alphas(angle, pixels)
    basis_vector = (np.cos(angle), np.sin(angle))
    number_of_alphas = len(alphas)
    radons = np.zeros(number_of_alphas)
    for i in range(number_of_alphas):
        radons[i] = integrate_for_radon(gamma_vector, gamma_der_vector, alphas[i], basis_vector)
    return radons


def integrate_for_radon(gamma_vector, gamma_der_vector, alpha, basis_vector):
    number_of_points = len(gamma_vector) - 1
    integral = 0
    for i in range(number_of_points):
        integral += (integrand(gamma_vector[i], gamma_der_vector[i], basis_vector, alpha)
                     + integrand(gamma_vector[i + 1], gamma_der_vector[i + 1], basis_vector, alpha)) / (
                            2 * number_of_points)
    return integral


def integrand(gamma_value, gamma_der_value, basis_vector, alpha):
    basis_vector_orthogonal = [-basis_vector[1], basis_vector[0]]
    if np.dot(gamma_value, basis_vector_orthogonal) - alpha < 0:
        return 0
    return -np.dot(gamma_der_value, basis_vector)
