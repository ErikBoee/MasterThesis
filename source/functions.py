import numpy as np


def theta(tau):
    return 2 * np.pi * tau + np.sin(16 * np.pi * tau)


def theta_ref(tau):
    return 2 * np.pi * tau


def der_theta(tau):
    return 4 * np.pi


def cos_theta(x):
    return np.cos(theta(x))


def sin_theta(x):
    return np.sin(theta(x))


def gamma(t_n, i, p, L):
    x_integral = 0
    y_integral = 0
    n = len(t_n) - 1
    assert (i <= n)
    for j in range(i):
        x_integral += (cos_theta(t_n[j]) + cos_theta(t_n[j + 1])) / 2 * n
        y_integral += (sin_theta(t_n[j]) + sin_theta(t_n[j + 1])) / 2 * n
    integral_values = np.array([x_integral, y_integral])
    return np.add(p, np.multiply(L, integral_values))


def der_gamma(t_n, i, L):
    return np.multiply(L, [cos_theta(t_n[i]), sin_theta(t_n[i])])


def energy_function(t_n):
    integral = 0
    n = len(t_n) - 1
    for i in range(n-1):
        integral += n / 2 * (((theta(t_n[i + 1]) - theta(t_n[i])) - (theta_ref(t_n[i + 1]) - theta_ref(t_n[i]))) ** 2 +
                             ((theta(t_n[i + 2]) - theta(t_n[i + 1])) - (
                                     theta_ref(t_n[i + 2]) - theta_ref(t_n[i + 1]))) ** 2)
    integral += n * ((theta(t_n[n]) - theta(t_n[n-1])) - (theta_ref(t_n[n]) - theta_ref(t_n[n-1]))) ** 2
    return integral

def radon_transform(gamma, angle, pixels, t_n):
    length_to_cross_screen = int(pixels / (np.max([np.cos(angle), np.sin(angle)])))
    alphas = np.linspace(0, length_to_cross_screen, length_to_cross_screen + 1)
    basis_vector = (np.cos(angle), np.sin(angle))
    radons = []
    for alpha in alphas:
        radons.append(integrate.quad(lambda x: integrand(x, basis_vector, alpha), 0, 1)[0])
    return radons, length_to_cross_screen

def integrand(t, basis_vector, alpha):
    if np.dot(gamma(t), basis_vector) - alpha < 0:
        return 0
    basis_vector_orthogonal = [-basis_vector[1], basis_vector[0]]
    return np.dot(der_gamma(t), basis_vector_orthogonal)
