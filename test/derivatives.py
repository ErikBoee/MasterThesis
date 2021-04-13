from constants import N_TIME, PIXELS, DELTA
import functions as func
import numpy as np
from numba import njit


def d_diff_length(gamma_vector, gamma_der_vector, angle, pixels, length,
                  differences_from_exact, point):
    alphas = func.get_alphas(angle, pixels)
    basis_vector = np.array([np.cos(angle), np.sin(angle)])
    basis_vector_orthogonal = np.array([-basis_vector[1], basis_vector[0]])
    number_of_alphas = len(alphas)
    derivatives_alpha = np.zeros(pixels)
    for i in range(number_of_alphas):
        derivative_vector = der_radon_given_alpha_length(gamma_vector, alphas[i], basis_vector,
                                                         basis_vector_orthogonal,
                                                         gamma_der_vector, length, point)
        derivatives_alpha[i] = func.trapezoidal_rule(derivative_vector, 1, 0)
    differences_from_exact = adjust_differences_from_exact_trapezoidal(differences_from_exact)

    return np.dot(differences_from_exact, derivatives_alpha)


def d_diff_point(gamma_vector, gamma_der_vector, angle, pixels,
                 differences_from_exact):
    alphas = func.get_alphas(angle, pixels)
    basis_vector = np.array([np.cos(angle), np.sin(angle)])
    basis_vector_orthogonal = np.array([-basis_vector[1], basis_vector[0]])
    number_of_alphas = len(alphas)
    derivatives_alpha = np.zeros((2, pixels))
    for i in range(number_of_alphas):
        derivative_vector = der_radon_given_alpha_point(gamma_vector, alphas[i], basis_vector,
                                                        basis_vector_orthogonal,
                                                        gamma_der_vector)
        derivatives_alpha[0][i] = func.trapezoidal_rule(derivative_vector[0], 1, 0)
        derivatives_alpha[1][i] = func.trapezoidal_rule(derivative_vector[1], 1, 0)
    differences_from_exact = adjust_differences_from_exact_trapezoidal(differences_from_exact)
    first_coordinate = np.dot(differences_from_exact, derivatives_alpha[0])
    second_coordinate = np.dot(differences_from_exact, derivatives_alpha[1])
    return np.array([first_coordinate, second_coordinate])


def adjust_differences_from_exact_trapezoidal(differences_from_exact):
    differences_from_exact[1:-1] = differences_from_exact[1:-1] / (PIXELS - 1)
    differences_from_exact[0] = differences_from_exact[0] / (2 * (PIXELS - 1))
    differences_from_exact[-1] = differences_from_exact[-1] / (2 * (PIXELS - 1))
    return differences_from_exact


def der_radon_given_alpha_point(gamma, alpha, basis_vector, basis_vector_orthogonal, gamma_der):
    derivatives = np.zeros((N_TIME + 1, 2))
    for i in range(N_TIME + 1):
        derivatives[i] = heaviside_cont_der(np.dot(gamma[i], basis_vector_orthogonal) - alpha,
                                            DELTA) * -np.dot(gamma_der[i], basis_vector) * basis_vector_orthogonal

    return derivatives.T


def der_radon_given_alpha_length(gamma, alpha, basis_vector, basis_vector_orthogonal, gamma_der, length, point):
    product_rule = np.zeros(N_TIME + 1)
    for i in range(N_TIME + 1):
        first_term_product_rule = heaviside_cont_der(np.dot(gamma[i], basis_vector_orthogonal) - alpha,
                                                     DELTA) * -np.dot(
            gamma_der[i], basis_vector) * np.dot(gamma[i] - point, basis_vector_orthogonal) / length
        second_term_product_rule = func.heaviside_cont_analytic(np.dot(gamma[i], basis_vector_orthogonal) - alpha,
                                                       DELTA) * -np.dot(
            gamma_der[i], basis_vector) / length

        product_rule[i] = first_term_product_rule + second_term_product_rule

    return product_rule


@njit
def der_radon_transform_point(gamma_vector, gamma_der_vector, angle, pixels):
    alphas = func.get_alphas(angle, pixels)
    basis_vector = np.array([np.cos(angle), np.sin(angle)])
    number_of_alphas = len(alphas)
    radons = np.zeros(number_of_alphas)
    for i in range(number_of_alphas):
        radons[i] = func.integrate_for_radon(gamma_vector, gamma_der_vector, alphas[i], basis_vector)
    return radons


@njit
def der_radon_diff_theta_given_alpha(alpha, basis_vector, basis_vector_orthogonal, entire_gamma, entire_gamma_der,
                                     gamma_diff_theta,
                                     der_gamma_diff_theta):
    derivatives = np.zeros(N_TIME)
    for i in range(N_TIME):
        if i == 0:
            derivatives[0] += func.heaviside_cont_analytic(np.dot(entire_gamma[i], basis_vector_orthogonal) - alpha,
                                                  DELTA) * -np.dot(
                der_gamma_diff_theta[i], basis_vector) / (2 * N_TIME)
            derivatives[0] += func.heaviside_cont_analytic(np.dot(entire_gamma[-1], basis_vector_orthogonal) - alpha,
                                                  DELTA) * -np.dot(
                der_gamma_diff_theta[i], basis_vector) / (2 * N_TIME)
        else:
            derivatives[i] += func.heaviside_cont_analytic(np.dot(entire_gamma[i], basis_vector_orthogonal) - alpha,
                                                  DELTA) * -np.dot(
                der_gamma_diff_theta[i], basis_vector) / N_TIME

        for j in range(N_TIME + 1):
            derivatives[i] += heaviside_cont_der(np.dot(entire_gamma[j], basis_vector_orthogonal) - alpha, DELTA) * \
                              np.dot(gamma_diff_theta[j, i], basis_vector_orthogonal) * -np.dot(entire_gamma_der[j],
                                                                                                basis_vector) / (
                                      N_TIME * (1 + int(j == 0 or j == N_TIME)))

    return derivatives


@njit
def heaviside_cont_der(x, delta):
    if -delta / 2 < x < delta / 2:
        return 1 / delta
    return 0.0
