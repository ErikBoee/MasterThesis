from copy import deepcopy

import numpy as np

import constants as const
import derivatives as der
import functions as func
import matplotlib.pyplot as plt


class OptimizationObjectGD:

    def __init__(self, init_theta, init_length, init_point, theta_ref, gamma_ref,
                 angle_to_exact_radon, beta, lamda, c, tau, max_iterator):
        self.theta = init_theta
        self.length = init_length
        self.point = init_point
        self.theta_ref = theta_ref
        self.gamma_ref = gamma_ref
        self.angle_to_alphas_and_exact_radon = OptimizationObjectGD.calculate_alphas_for_dict(angle_to_exact_radon)
        self.update_gamma()
        self.beta = beta
        self.lamda = lamda
        self.c = c
        self.tau = tau
        self.max_iterator = max_iterator

    def armijo_backtracking(self, gradient, gradient_theta, gradient_length, gradient_point, step_size):
        m = -np.linalg.norm(gradient)
        current_obj_function = self.objective_function()
        self.theta[:-1] -= step_size * gradient_theta
        self.theta[-1] = self.theta[0] + 2 * np.pi
        self.point -= step_size * gradient_point
        self.length -= step_size * gradient_length
        self.update_gamma()
        new_objective_function = self.objective_function()
        while current_obj_function - new_objective_function < -m * step_size * self.c and step_size > 10 ** (-16):
            self.theta[:-1] += (1 - self.tau) * step_size * gradient_theta
            self.theta[-1] = self.theta[0] + 2 * np.pi
            self.point += (1 - self.tau) * step_size * gradient_point
            self.length += (1 - self.tau) * step_size * gradient_length
            step_size = self.tau * step_size
            self.update_gamma()
            new_objective_function = self.objective_function()
        return step_size

    def gradient_descent(self):
        func.draw_boundary(self.gamma, self.gamma_ref, -1, const.PIXELS)
        iterator = 0
        iterator = self.gradient_descent_to_convergence_point(iterator)
        print(const.ITERATOR, iterator)
        quadratic_penalty = const.PENALTY_TOL + 1
        while quadratic_penalty > const.PENALTY_TOL and self.lamda < const.MAX_LAMDA:
            print("Lambda: ", self.lamda)
            iterator = self.gradient_descent_to_convergence_m(iterator)
            print(const.ITERATOR, iterator)
            self.lamda *= 10
            quadratic_penalty = self.quadratic_penalty_term()
            iterator = 0
        return self.theta, self.length, self.point, iterator, self.objective_function()

    def gradient_descent_to_convergence_m(self, iterator):
        former_obj = -2 * const.TOL
        count = 0
        while (abs(former_obj - self.objective_function()) > const.TOL or count < 3) and iterator < self.max_iterator:
            former_obj = self.objective_function()
            gradient_theta_num = self.numerical_gradient_theta()
            gradient_length_num = self.numerical_gradient_length()
            gradient_point_num = self.numerical_gradient_point()
            self.inner_loop_update_gradient_descent(gradient_theta_num, gradient_length_num, gradient_point_num,
                                                    iterator)

            if abs(former_obj - self.objective_function()) <= const.TOL:
                count += 1
            iterator += 1
        print("Finished optimization m")
        return iterator

    def gradient_descent_to_convergence_length(self, iterator):
        former_obj = -2 * const.TOL
        count = 0
        while (abs(former_obj - self.objective_function()) > const.TOL or count < 3) and iterator < self.max_iterator:
            former_obj = self.objective_function()
            gradient_theta_num = np.zeros(len(self.theta) - 1)
            gradient_length_num = self.numerical_gradient_length()
            gradient_point_num = np.zeros(len(self.point))
            self.inner_loop_update_gradient_descent(gradient_theta_num, gradient_length_num, gradient_point_num,
                                                    iterator)
            if abs(former_obj - self.objective_function()) <= const.TOL:
                count += 1
            iterator += 1
        print("Finished optimization length")
        return iterator

    def gradient_descent_to_convergence_point(self, iterator):
        former_obj = -2 * const.TOL
        count = 0
        while (abs(former_obj - self.objective_function()) > const.TOL or count < 3) and iterator < self.max_iterator:
            former_obj = self.objective_function()
            gradient_theta_num = np.zeros(len(self.theta) - 1)
            gradient_length_num = 0.0
            gradient_point_num = self.numerical_gradient_point()
            self.inner_loop_update_gradient_descent(gradient_theta_num, gradient_length_num, gradient_point_num,
                                                    iterator)
            if abs(former_obj - self.objective_function()) <= const.TOL:
                count += 1
            iterator += 1
        print("Finished optimization point")
        return iterator

    def inner_loop_update_gradient_descent(self, gradient_theta_num, gradient_length_num, gradient_point_num, iterator):
        gradient_num = np.concatenate((gradient_point_num, [gradient_length_num], gradient_theta_num),
                                      axis=0)
        step_size = const.STEPSIZE
        step_size = self.armijo_backtracking(gradient_num, gradient_theta_num, gradient_length_num,
                                             gradient_point_num, step_size)

        #self.display_information(iterator, step_size, gradient_num)

    def display_information(self, iterator, step_size, gradient_num):
        if iterator % 10 == 0:
            self.print_objective_information(iterator, step_size, gradient_num)
            #           self.compare_radon_transforms(iterator)

            if iterator % 10 == 0:
                func.draw_boundary(self.gamma, self.gamma_ref, iterator, const.PIXELS)

    def compare_radon_transforms(self, iterator):
        for angle, dictionary in self.angle_to_alphas_and_exact_radon.items():
            plt.plot(np.linspace(0, 1, const.PIXELS), dictionary[const.EXACT_RADON_TRANSFORM], label="exact")
            plt.plot(np.linspace(0, 1, const.PIXELS),
                     func.radon_transform(self.gamma, self.gamma_der, angle, const.PIXELS), label="calculated")
            plt.title(str(iterator))
            plt.legend()
            plt.show()

    def print_objective_information(self, iterator, step_size, gradient_num):
        print("Iterator:", iterator)
        print("Objective value: ", self.objective_function())
        print("Objective value first term: ", self.objective_function_first_term(0))
        print("Objective value energy term: ", self.objective_function_energy_term())
        print("Objective value penalty term: ", self.objective_function_penalty_term())
        print("Norm Gradient numerical: ", np.linalg.norm(gradient_num))
        print("Step size:", step_size)

    def update_gamma(self):
        self.gamma = func.calculate_entire_gamma_from_theta(self.theta, self.point, self.length)
        self.gamma_der = func.calculate_entire_gamma_der_from_theta(self.theta, self.length)

    def objective_function(self):
        obj = 0
        obj = self.objective_function_first_term(obj)
        obj += self.objective_function_energy_term()
        obj += self.objective_function_penalty_term()
        return obj

    def quadratic_penalty_term(self):
        cos_integral = func.trapezoidal_rule(np.cos(self.theta), 1, 0)
        sin_integral = func.trapezoidal_rule(np.sin(self.theta), 1, 0)
        return cos_integral ** 2 + sin_integral ** 2

    def objective_function_first_term(self, obj):
        for angle, dictionary in self.angle_to_alphas_and_exact_radon.items():
            difference_in_radon = self.calc_radon_transform_minus_exact(dictionary[const.EXACT_RADON_TRANSFORM], angle)
            difference_in_radon_squared = difference_in_radon ** 2
            l2_norm = func.trapezoidal_rule(difference_in_radon_squared, 1, 0)
            obj += l2_norm / 2.0
        return obj

    def objective_function_energy_term(self):
        return self.beta / 2.0 * self.energy_function()

    def objective_function_penalty_term(self):
        return self.lamda * (self.quadratic_penalty_term())

    def energy_function(self):
        der_theta = (self.theta[1:] - self.theta[:-1]) * const.N_TIME
        der_theta_ref = (self.theta_ref[1:] - self.theta_ref[:-1]) * const.N_TIME
        difference_theta_der = der_theta - der_theta_ref
        difference_theta_der_squared = difference_theta_der ** 2
        return func.trapezoidal_rule(difference_theta_der_squared, 1 - 1 / (2 * const.N_TIME),
                                     1 / (2 * const.N_TIME))

    @staticmethod
    def calculate_alphas_for_dict(angle_to_exact_radon):
        for angle, dictionary in angle_to_exact_radon.items():
            alphas = func.get_alphas(angle, const.PIXELS)
            dictionary["alphas"] = alphas
            angle_to_exact_radon[angle] = dictionary
        return angle_to_exact_radon

    def calc_radon_transform_minus_exact(self, exact_transform, angle):
        differences_from_exact = (
                func.radon_transform(self.gamma, self.gamma_der, angle, const.PIXELS) - exact_transform.T)
        return_vector = differences_from_exact[0]
        return return_vector

    def numerical_gradient_point(self):
        self.update_gamma()
        actual_gamma = deepcopy(self.gamma)
        actual_point = deepcopy(self.point)
        actual_gamma_der = deepcopy(self.gamma_der)
        former_obj = self.objective_function()
        self.point[0] += const.EPSILON
        self.gamma = func.calculate_entire_gamma_from_theta(self.theta, self.point, self.length)
        x_changed_obj = self.objective_function()
        self.point[0] -= const.EPSILON
        self.point[1] += const.EPSILON
        self.gamma = func.calculate_entire_gamma_from_theta(self.theta, self.point, self.length)
        y_changed_obj = self.objective_function()
        der_p_x = (x_changed_obj - former_obj) / const.EPSILON
        der_p_y = (y_changed_obj - former_obj) / const.EPSILON
        self.reset_gamma(actual_gamma, actual_gamma_der)
        self.point = actual_point
        return np.array([der_p_x, der_p_y])

    def numerical_gradient_theta(self):
        self.update_gamma()
        derivatives = np.zeros(const.N_TIME)
        actual_gamma = deepcopy(self.gamma)
        actual_gamma_der = deepcopy(self.gamma_der)
        actual_theta = deepcopy(self.theta)
        former_obj = self.objective_function()
        for i in range(const.N_TIME):
            derivatives[i] = self.numerical_derivative_theta_i(i, former_obj)
            self.theta = deepcopy(actual_theta)
        self.reset_gamma(actual_gamma, actual_gamma_der)
        return derivatives

    def numerical_derivative_theta_i(self, i, former_obj):
        coord = np.zeros(const.N_TIME + 1)
        if i == 0:
            coord[0] = const.EPSILON
            coord[-1] = const.EPSILON
        else:
            coord[i] = const.EPSILON
        self.theta += coord
        self.update_gamma()
        changed_obj = self.objective_function()
        return (changed_obj - former_obj) / const.EPSILON

    def numerical_gradient_length(self):
        self.update_gamma()
        actual_gamma = deepcopy(self.gamma)
        actual_gamma_der = deepcopy(self.gamma_der)
        actual_length = deepcopy(self.length)
        former_obj = self.objective_function()
        self.length += const.EPSILON
        self.update_gamma()
        changed_obj = self.objective_function()
        derivative = (changed_obj - former_obj) / const.EPSILON
        self.length = actual_length
        self.reset_gamma(actual_gamma, actual_gamma_der)
        return derivative

    def reset_gamma(self, actual_gamma, actual_gamma_der):
        self.gamma = actual_gamma
        self.gamma_der = actual_gamma_der

    # Analytical gradients

    def gradient_length(self):
        self.update_gamma()
        return self.functional_diff_length()

    def gradient_point(self):
        self.update_gamma()
        return self.functional_diff_point()

    def gradient_theta(self):
        self.update_gamma()
        der_d_diff_theta = self.der_d_diff_theta()
        return der_d_diff_theta + self.beta / 2 * self.der_energy_func_theta() + \
               self.der_quadratic_penalty_term_diff_theta()

    def der_quadratic_penalty_term_diff_theta(self):
        sin_sum = func.trapezoidal_rule(np.sin(self.theta), 1, 0)
        cos_sum = func.trapezoidal_rule(np.cos(self.theta), 1, 0)
        theta_cos = 2 * self.lamda * np.cos(self.theta[:-1]) * sin_sum / const.N_TIME
        theta_sin = -2 * self.lamda * np.sin(self.theta[:-1]) * cos_sum / const.N_TIME
        return theta_cos + theta_sin

    @staticmethod
    def calc_derivative_d_diff_theta_given_alphas(differences_from_exact, alphas, basis_vector,
                                                  basis_vector_orthogonal, entire_gamma, entire_gamma_der,
                                                  gamma_diff_theta, der_gamma_diff_theta):

        derivatives = np.zeros(const.N_TIME)

        for i in range(const.PIXELS):
            derivative_for_alpha = np.multiply(differences_from_exact[i],
                                               der.der_radon_diff_theta_given_alpha(alphas[i],
                                                                                    basis_vector,
                                                                                    basis_vector_orthogonal,
                                                                                    entire_gamma,
                                                                                    entire_gamma_der,
                                                                                    gamma_diff_theta,
                                                                                    der_gamma_diff_theta))

            if i == 0 or i == const.PIXELS - 1:
                derivatives += derivative_for_alpha / (2 * (const.PIXELS - 1))
            else:
                derivatives += derivative_for_alpha / (const.PIXELS - 1)
        return derivatives

    def der_energy_func_theta(self):
        derivatives = np.zeros(const.N_TIME)
        derivatives[0] = const.N_TIME * (2 * self.theta[0] - self.theta_ref[0]
                                         - self.theta[1] + self.theta_ref[1] + 2 * np.pi
                                         - self.theta_ref[const.N_TIME] - self.theta[const.N_TIME - 1] + self.theta_ref[
                                             const.N_TIME - 1])

        derivatives[1] = const.N_TIME * (3 * self.theta[1] - 3 * self.theta_ref[1]
                                         - self.theta[0] + self.theta_ref[0]
                                         - 2 * self.theta[2] + 2 * self.theta_ref[2])

        derivatives[2:const.N_TIME - 1] = 2 * const.N_TIME * (
                2 * self.theta[2:const.N_TIME - 1] - 2 * self.theta_ref[2:const.N_TIME - 1]
                - self.theta[1:const.N_TIME - 2] + self.theta_ref[1:const.N_TIME - 2]
                - self.theta[3:const.N_TIME] + self.theta_ref[3:const.N_TIME])
        derivatives[const.N_TIME - 1] = const.N_TIME * (
                3 * self.theta[const.N_TIME - 1] - 3 * self.theta_ref[const.N_TIME - 1]
                - self.theta[const.N_TIME] + self.theta_ref[const.N_TIME]
                - 2 * self.theta[const.N_TIME - 2] + 2 * self.theta_ref[const.N_TIME - 2])

        return derivatives

    def functional_diff_point(self):
        derivative = np.array([0.0, 0.0])
        for angle, dictionary in self.angle_to_alphas_and_exact_radon.items():
            differences_from_exact = self.calc_radon_transform_minus_exact(dictionary[const.EXACT_RADON_TRANSFORM],
                                                                           angle)
            element = der.d_diff_point(self.gamma, self.gamma_der, angle, const.PIXELS, differences_from_exact)
            derivative += element
        return derivative

    def der_d_diff_theta_given_angle(self, angle, alphas,
                                     exact_transform):
        differences_from_exact = self.calc_radon_transform_minus_exact(exact_transform, angle)
        basis_vector = np.array([np.cos(angle), np.sin(angle)])
        basis_vector_orthogonal = np.array([-basis_vector[1], basis_vector[0]])
        gamma_diff_theta = func.gamma_diff_theta(self.theta, self.length)
        gamma_der_diff_theta = func.der_gamma_diff_theta(self.theta, self.length)
        derivatives = OptimizationObjectGD.calc_derivative_d_diff_theta_given_alphas(differences_from_exact, alphas,
                                                                                     basis_vector,
                                                                                     basis_vector_orthogonal, self.gamma,
                                                                                     self.gamma_der,
                                                                                     gamma_diff_theta,
                                                                                     gamma_der_diff_theta)
        return derivatives

    def der_d_diff_theta(self):
        derivatives = np.zeros(const.N_TIME)
        for angle, dictionary in self.angle_to_alphas_and_exact_radon.items():
            derivatives += self.der_d_diff_theta_given_angle(angle, dictionary[const.ALPHAS],
                                                             dictionary[const.EXACT_RADON_TRANSFORM])
        return derivatives

    def functional_diff_length(self):
        derivative = 0
        for angle, dictionary in self.angle_to_alphas_and_exact_radon.items():
            differences_from_exact = self.calc_radon_transform_minus_exact(dictionary[const.EXACT_RADON_TRANSFORM],
                                                                           angle)
            derivative += der.d_diff_length(self.gamma, self.gamma_der, angle, const.PIXELS,
                                            self.length, differences_from_exact, self.point)
        return derivative
