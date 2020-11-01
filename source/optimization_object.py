import numpy as np
from source.constants import N_TIME, PIXELS, EXACT_RADON_TRANSFORM, ALPHAS, EPSILON, STEPSIZE, TOL
import source.functions as func
from numba import njit
from copy import deepcopy


class QuadraticPenalty:

    def __init__(self, init_theta, init_length, init_point, theta_ref, gamma_ref, angle_to_exact_radon, beta):
        self.theta = init_theta
        self.length = init_length
        self.point = init_point
        self.theta_ref = theta_ref
        self.gamma_ref = gamma_ref
        self.angle_to_alphas_and_exact_radon = QuadraticPenalty.calculate_alphas_for_dict(angle_to_exact_radon)
        self.gamma = func.calculate_entire_gamma_from_theta(self.theta, self.point, self.length)
        self.gamma_der = func.calculate_entire_gamma_der_from_theta(self.theta, self.length)
        self.beta = beta
        self.lamda = 1500
        self.c = 0.5
        self.tau = 0.5

    def armijo_backtracking(self, gradient, gradient_theta, gradient_length, gradient_point, step_size):
        m = -np.linalg.norm(gradient)
        current_obj_function = self.objective_function()
        self.theta[:-1] -= step_size * gradient_theta
        self.theta[-1] = self.theta[0] + 2 * np.pi
        self.point -= step_size * gradient_point
        self.length -= step_size * gradient_length
        while current_obj_function - self.objective_function() < -m * step_size * self.c:
            step_size = self.tau*step_size
            self.theta[:-1] += step_size * gradient_theta
            self.theta[-1] = self.theta[0] + 2 * np.pi
            self.point += step_size * gradient_point
            self.length += step_size * gradient_length
        print(step_size)
        return self.theta, self.length, self.point

    def gradient_descent(self):
        step_size = STEPSIZE
        print(self.gradient_point())
        print(self.gradient_length())
        print(self.gradient_theta())
        gradient = np.concatenate((self.gradient_point(), [self.gradient_length()], self.gradient_theta()),
                                  axis=0)
        print(gradient)
        iterator = 0
        while np.linalg.norm(gradient) > TOL and iterator < 10:
            gradient_theta = self.gradient_theta()
            gradient_length = self.gradient_length()
            gradient_point = self.gradient_point()
            gradient = np.concatenate((gradient_point, [gradient_length], gradient_theta),
                                      axis=0)
            self.theta, self.length, self.point = self.armijo_backtracking(gradient, gradient_theta, gradient_length,
                                                                          gradient_point, step_size)
            #self.length -= step_size * gradient_length
            #self.point -= step_size * gradient_point
            #self.theta[-1] = self.theta[0] + 2 * np.pi
            self.gamma = func.calculate_entire_gamma_from_theta(self.theta, self.point, self.length)
            self.gamma_der = func.calculate_entire_gamma_der_from_theta(self.theta, self.length)

            func.draw_boundary(self.gamma, self.gamma_ref, iterator, PIXELS)
            iterator += 1
            print(iterator)

    def gradient_length(self):
        return self.der_d_diff_length()

    def gradient_point(self):
        return self.der_first_term_diff_p()

    def gradient_theta(self):
        return self.der_d_diff_theta() + self.beta/2*self.der_energy_func_theta() + self.der_quadratic_penalty_term_diff_theta()

    def objective_function(self):
        obj = 0
        obj = self.objective_function_first_term(obj)
        obj += self.beta / 2.0 * self.energy_function()
        obj += self.lamda * (self.quadratic_penalty_term())
        return obj

    def quadratic_penalty_term(self):
        cos_integral = func.trapezoidal_normalized_to_unit_interval(np.cos(self.theta))
        sin_integral = func.trapezoidal_normalized_to_unit_interval(np.sin(self.theta))
        return cos_integral ** 2 + sin_integral ** 2

    def objective_function_first_term(self, obj):
        for angle, dictionary in self.angle_to_alphas_and_exact_radon.items():
            difference_in_radon = self.calc_radon_transform_minus_exact(dictionary[EXACT_RADON_TRANSFORM], angle)
            difference_in_radon_squared = difference_in_radon ** 2
            l2_norm = func.trapezoidal_normalized_to_unit_interval(difference_in_radon_squared)
            obj += l2_norm / 2.0
        return obj

    def energy_function(self):
        der_theta = (self.theta[1:] - self.theta[:-1]) * N_TIME
        der_theta_ref = (self.theta_ref[1:] - self.theta_ref[:-1]) * N_TIME
        difference_theta_der = der_theta - der_theta_ref
        difference_theta_der_squared = difference_theta_der ** 2
        return func.trapezoidal_normalized_to_unit_interval(difference_theta_der_squared)

    @staticmethod
    def calculate_alphas_for_dict(angle_to_exact_radon):
        for angle, dictionary in angle_to_exact_radon.items():
            alphas = func.get_alphas(angle, PIXELS)
            dictionary["alphas"] = alphas
            angle_to_exact_radon[angle] = dictionary
        return angle_to_exact_radon

    def der_gamma_diff_theta(self):
        return np.multiply(self.length, np.array([-np.sin(self.theta), np.cos(self.theta)]).T)

    def der_d_diff_theta(self):
        derivatives = np.zeros(N_TIME)
        for angle, dictionary in self.angle_to_alphas_and_exact_radon.items():
            derivatives += self.der_d_diff_theta_given_angle(angle, dictionary[ALPHAS],
                                                             dictionary[EXACT_RADON_TRANSFORM])
        return derivatives

    def der_d_diff_length(self):
        derivative = 0
        for angle, dictionary in self.angle_to_alphas_and_exact_radon.items():
            differences_from_exact = self.calc_radon_transform_minus_exact(dictionary[EXACT_RADON_TRANSFORM], angle)
            derivative += self.der_d_diff_length_given_angle(angle, differences_from_exact)
        return derivative

    def der_d_diff_length_given_angle(self, angle, differences_from_exact):
        radon_transform_diff_length = func.radon_transform(self.gamma, self.gamma_der, angle, PIXELS) / self.length
        return np.dot(differences_from_exact, radon_transform_diff_length)

    def der_d_diff_theta_given_angle(self, angle, alphas,
                                     exact_transform):
        differences_from_exact = self.calc_radon_transform_minus_exact(exact_transform, angle)
        basis_vector = np.array([np.cos(angle), np.sin(angle)])
        basis_vector_orthogonal = np.array([-basis_vector[1], basis_vector[0]])
        derivatives = QuadraticPenalty.calc_derivative_d_diff_theta_given_alphas(differences_from_exact, alphas,
                                                                                 basis_vector,
                                                                                 basis_vector_orthogonal, self.gamma,
                                                                                 self.der_gamma_diff_theta())

        return derivatives

    @staticmethod
    def calc_derivative_d_diff_theta_given_alphas(differences_from_exact, alphas, basis_vector,
                                                  basis_vector_orthogonal, entire_gamma, der_gamma_diff_theta):

        derivatives = np.zeros(N_TIME)
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
    @njit
    def der_radon_diff_theta_given_alpha(alpha, basis_vector, basis_vector_orthogonal, entire_gamma,
                                         der_gamma_diff_theta):
        derivatives = np.zeros(N_TIME)
        for i in range(N_TIME):
            if np.dot(entire_gamma[i], basis_vector_orthogonal) - alpha < 0:
                derivatives[i] = 0
            else:
                derivatives[i] = -np.dot(der_gamma_diff_theta[i], basis_vector) / N_TIME
        return derivatives

    def calc_radon_transform_minus_exact(self, exact_transform, angle):
        return (func.radon_transform(self.gamma, self.gamma_der, angle, PIXELS) - exact_transform.T)[0]

    def der_energy_func_theta(self):
        derivatives = np.zeros(N_TIME)
        derivatives[0] = N_TIME * (2 * self.theta[0] - self.theta_ref[0]
                                   - self.theta[1] + self.theta_ref[1] + 2 * np.pi
                                   - self.theta_ref[N_TIME] - self.theta[N_TIME - 1] + self.theta_ref[N_TIME - 1])

        derivatives[1] = N_TIME * (3 * self.theta[1] - 3 * self.theta_ref[1]
                                   - self.theta[0] + self.theta_ref[0]
                                   - 2 * self.theta[2] + 2 * self.theta[2])

        derivatives[2:N_TIME - 1] = 2 * N_TIME * (2 * self.theta[2:N_TIME - 1] - 2 * self.theta_ref[2:N_TIME - 1]
                                                  - self.theta[1:N_TIME - 2] + self.theta_ref[1:N_TIME - 2]
                                                  - self.theta[3:N_TIME] + self.theta[3:N_TIME])
        derivatives[N_TIME - 1] = N_TIME * (3 * self.theta[N_TIME - 1] - 3 * self.theta_ref[N_TIME - 1]
                                            - self.theta[N_TIME] + self.theta_ref[N_TIME]
                                            - 2 * self.theta[N_TIME - 2] + 2 * self.theta[N_TIME - 2])

        return derivatives

    def der_quadratic_penalty_term_diff_theta(self):
        sin_sum = (np.sum(np.sin(self.theta[:-1])) / N_TIME)
        cos_sum = (np.sum(np.cos(self.theta[:-1])) / N_TIME)
        theta_cos = 2 * self.lamda * np.cos(self.theta[:-1]) * sin_sum
        theta_sin = -2 * self.lamda * np.sin(self.theta[:-1]) * cos_sum
        return theta_cos + theta_sin

    def der_first_term_diff_p(self):
        actual_gamma = deepcopy(self.gamma)
        former_obj = 0
        former_obj = self.objective_function_first_term(former_obj)
        x_coord_eps = EPSILON * np.array([1, 0])
        y_coord_eps = EPSILON * np.array([0, 1])
        self.gamma = func.calculate_entire_gamma_from_theta(self.theta, self.point + x_coord_eps, self.length)
        x_changed_obj = 0
        x_changed_obj = self.objective_function_first_term(x_changed_obj)
        der_p_x = (x_changed_obj - former_obj) / EPSILON
        self.gamma = func.calculate_entire_gamma_from_theta(self.theta, self.point + y_coord_eps, self.length)
        y_changed_obj = 0
        y_changed_obj = self.objective_function_first_term(y_changed_obj)
        der_p_y = (y_changed_obj - former_obj) / EPSILON
        self.gamma = actual_gamma
        return np.array([der_p_x, der_p_y])
