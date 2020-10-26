import numpy as np
from source.constants import N_TIME, PIXELS
import source.functions as func


class QuadraticPenalty():
    def __init__(self, init_theta, init_length, init_point, theta_ref, exact_radon_transforms, angles):
        self.theta = init_theta
        self.length = init_length
        self.point = init_point
        self.theta_ref = theta_ref
        self.exact_radon_transforms = exact_radon_transforms
        self.angles = angles

    def objective_function(self):
        return

    def der_gamma_diff_theta(self):
        return np.multiply(self.length, np.array([-np.sin(self.theta), np.cos(self.theta)]).T)

    def der_d_diff_theta_given_angle(self, angle, entire_gamma, entire_gamma_der, der_gamma_diff_theta,
                                     exact_transform):
        alphas = func.get_alphas(angle, PIXELS)
        derivatives = np.zeros(N_TIME - 1)
        differences_from_exact = self.calc_radon_transform_minus_exact(exact_transform, angle, entire_gamma,
                                                                       entire_gamma_der)
        basis_vector = np.array([np.cos(angle), np.sin(angle)])
        basis_vector_orthogonal = np.array([-basis_vector[1], basis_vector[0]])
        for i in range(PIXELS):
            derivative_for_alpha = np.multiply(differences_from_exact[i],
                                               QuadraticPenalty.der_radon_diff_theta_given_alpha(alphas[i],
                                                                                                 basis_vector,
                                                                                                 basis_vector_orthogonal,
                                                                                                 entire_gamma,
                                                                                                 der_gamma_diff_theta))
            if i == 0 or i == PIXELS - 1:
                derivatives += derivative_for_alpha / (2 * PIXELS)
            else:
                derivatives += derivative_for_alpha / PIXELS

        return derivatives

    @staticmethod
    def der_radon_diff_theta_given_alpha(alpha, basis_vector, basis_vector_orthogonal, entire_gamma,
                                         der_gamma_diff_theta):
        derivatives = np.zeros(N_TIME - 1)
        for i in range(N_TIME - 1):
            if np.dot(entire_gamma[i], basis_vector_orthogonal) - alpha < 0:
                derivatives[i] = 0
            else:
                derivatives[i] = -np.dot(der_gamma_diff_theta[i], basis_vector) / N_TIME
        return derivatives

    def calc_radon_transform_minus_exact(self, exact_transform, angle, gamma_vector, gamma_der_vector):
        return func.radon_transform(gamma_vector, gamma_der_vector, angle, PIXELS) - exact_transform

    def der_energy_func_theta(self):
        return
