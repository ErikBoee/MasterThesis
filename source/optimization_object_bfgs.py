from copy import deepcopy

import numpy as np

import constants as const
import functions as func
import matplotlib.pyplot as plt
import cv2
import os


class OptimizationObjectBFGS:

    def __init__(self, init_theta, init_length, init_point, theta_ref, gamma_ref,
                 angle_to_exact_radon, beta, lamda, c_1, c_2, tau, max_iterator, image_frequency):
        self.theta = init_theta
        self.length = init_length
        self.point = init_point
        self.theta_ref = theta_ref
        self.gamma_ref = gamma_ref
        self.angle_to_alphas_and_exact_radon = OptimizationObjectBFGS.calculate_alphas_for_dict(angle_to_exact_radon)
        self.update_gamma()
        self.beta = beta
        self.lamda = lamda
        self.c_1 = c_1
        self.c_2 = c_2
        self.tau = tau
        self.max_iterator = max_iterator
        self.image_frequency = image_frequency

    def wolfe_conditions(self, search_direction, gradient, step_size):
        max_iter = 100
        alpha_i_1 = 0
        alpha_max = step_size
        i = 1
        alpha_i = alpha_max / 2
        m = search_direction.T @ gradient
        former_obj = self.objective_function(self.theta, self.length, self.point)
        obj_i_1 = former_obj
        while i < max_iter and alpha_i < alpha_max:
            theta_i, length_i, point_i = self.update_variables(alpha_i, search_direction)
            obj_i = self.objective_function(theta_i, length_i, point_i)
            if obj_i > former_obj + self.c_1 * m * alpha_i or (i > 1 and obj_i >= obj_i_1):
                return self.zoom(alpha_i_1, alpha_i, former_obj, search_direction, m)
            gradient_i = self.get_gradient(theta_i, length_i, point_i)
            m_i = search_direction.T @ gradient_i
            if abs(m_i) <= self.c_2 * m:
                return alpha_i
            if m_i >= 0:
                return self.zoom(alpha_i, alpha_i_1, former_obj, search_direction, m)
            obj_i_1 = obj_i
            alpha_i_1 = alpha_i
            alpha_i = (alpha_i + alpha_max) / 2
        return alpha_i

    def update_variables(self, alpha, search_direction):
        theta = np.zeros(len(self.theta))
        theta[:-1] = deepcopy(self.theta[:-1]) + alpha * search_direction[3:]
        theta[-1] = theta[0] + 2 * np.pi
        point = deepcopy(self.point) + alpha * search_direction[:2]
        length = self.length + alpha * search_direction[2]
        return theta, length, point

    def zoom(self, alpha_low, alpha_high, former_obj, search_direction, m):
        max_iter = 100
        j = 1
        alpha_j = (alpha_high + alpha_low) / 2
        while j < max_iter:
            theta_j, length_j, point_j = self.update_variables(alpha_j, search_direction)
            if self.objective_function(theta_j, length_j, point_j) > former_obj + self.c_1 * m * alpha_j:
                alpha_high = alpha_j
            else:
                gradient_j = self.get_gradient(theta_j, length_j, point_j)
                m_j = search_direction.T @ gradient_j
                if abs(m_j) <= -self.c_2 * m:
                    return alpha_j
                if m_j * (alpha_high - alpha_low) >= 0:
                    alpha_high = alpha_low
                alpha_low = alpha_j
            alpha_j = (alpha_high + alpha_low) / 2
            j += 1
        return alpha_j

    def get_gradient(self, theta, length, point):
        gradient_theta_num = self.numerical_gradient_theta(theta, length, point)
        gradient_length_num = self.numerical_gradient_length(theta, length, point)
        gradient_point_num = self.numerical_gradient_point(theta, length, point)
        gradient_num = np.concatenate((gradient_point_num, [gradient_length_num], gradient_theta_num),
                                      axis=0)
        return gradient_num

    @staticmethod
    def get_variables(theta, length, point):
        return np.concatenate((point, [length], theta[:-1]),
                              axis=0)

    def bfgs(self, folder_path):
        os.chdir(folder_path)
        func.draw_boundary(self.gamma, self.gamma_ref, -1, const.PIXELS)
        iterator = 0
        total_iterator = []
        quadratic_penalty = const.PENALTY_TOL + 1
        for j in range(const.NUMBER_OF_FULL_LOOPS):
            while quadratic_penalty > const.PENALTY_TOL and self.lamda < const.MAX_LAMDA:
                iterator = self.bfgs_to_convergence_m(iterator, j)
                total_iterator.append([iterator, self.lamda])
                self.lamda *= 10
                quadratic_penalty = self.quadratic_penalty_term(self.theta)
                iterator = 0

            self.create_dict_and_save(iterator, j)
            self.theta_ref = deepcopy(self.theta)
            self.lamda = const.LAMDA
            quadratic_penalty = const.PENALTY_TOL + 1
            # self.beta = self.beta*0.5
        return self.theta, self.length, self.point, total_iterator, \
               self.objective_function(self.theta, self.length, self.point)

    def create_dict_and_save(self, i, j):
        self.update_gamma()
        boundary_image = func.get_boundary_image(self.gamma, self.gamma_ref, const.PIXELS)
        cv2.imwrite("j_" + str(j) + "_lambda_" + str(self.lamda) + "_i_" + str(i) + ".png", boundary_image)
        iterator_dict = self.create_information_dictionary()
        np.save("j_" + str(j) + "_lambda_" + str(self.lamda) + "_i_" + str(i), iterator_dict, allow_pickle=True)

    def create_information_dictionary(self):
        dikt = {}
        dikt["Objective function first term"] = self.objective_function_first_term(0, self.theta, self.length,
                                                                                   self.point)
        dikt["Objective function energy term"] = self.objective_function_energy_term(self.theta)
        dikt["Objective function penalty term"] = self.objective_function_penalty_term(self.theta)
        dikt["Lamda"] = self.lamda
        dikt["Norm of gradient"] = np.linalg.norm(self.get_gradient(self.theta, self.length, self.point))
        dikt["Theta"] = self.theta
        dikt["Length"] = self.length
        dikt["Point"] = self.point
        dikt["Mobius"] = self.mobius_energy(func.calculate_entire_gamma_from_theta(self.theta, self.point, self.length))
        return dikt

    def bfgs_to_convergence_m(self, iterator, j):
        former_obj = -2 * const.TOL
        count = 0
        beta_k_inv = np.eye(len(self.theta) + 2)
        while (abs(former_obj - self.objective_function(self.theta, self.length, self.point)) / former_obj
               > const.TOL or count < 5) \
                and iterator < self.max_iterator:
            former_obj = self.objective_function(self.theta, self.length, self.point)
            gradient_num = self.get_gradient(self.theta, self.length, self.point)
            beta_k_inv, self.theta, self.length, self.point = \
                self.inner_loop_update_bfgs(gradient_num, beta_k_inv)

            if abs(former_obj - self.objective_function(self.theta, self.length, self.point)) <= const.TOL:
                count += 1
            iterator += 1
            if iterator % self.image_frequency == 0:
                self.create_dict_and_save(iterator, j)
                # self.update_gamma()
                # func.draw_boundary(self.gamma, self.gamma_ref, iterator, const.PIXELS)
                # self.print_objective_information(iterator, gradient_num)
        print("Finished optimization m")
        return iterator

    def inner_loop_update_bfgs(self, gradient_num, beta_k_inv):
        Identity = np.eye(len(gradient_num))
        step_size = const.STEPSIZE * 2
        search_direction = -beta_k_inv @ gradient_num
        step_size = self.wolfe_conditions(search_direction, gradient_num, step_size)
        theta_next, length_next, point_next = self.update_variables(step_size, search_direction)
        s_k = OptimizationObjectBFGS.get_variables(theta_next, length_next,
                                                   point_next) - OptimizationObjectBFGS.get_variables(self.theta,
                                                                                                      self.length,
                                                                                                      self.point)
        next_gradient = self.get_gradient(theta_next, length_next, point_next)
        y_k = next_gradient - gradient_num
        if (y_k.T @ s_k) < const.STEPSIZE_TOL:
            print("Too short stepsize")
            return beta_k_inv, theta_next, length_next, point_next
        else:
            rho_k = 1 / (y_k.T @ s_k)
            beta_k_plus_1_inv = ((Identity - rho_k * np.outer(s_k, y_k.T)) @
                                 beta_k_inv @ (Identity - rho_k * np.outer(y_k, s_k.T))
                                 + rho_k * np.outer(s_k, s_k.T))
            return beta_k_plus_1_inv, theta_next, length_next, point_next

    def display_information(self, iterator, step_size, gradient_num):
        if iterator % 10 == 0:
            self.print_objective_information(iterator, step_size
                                             )
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

    def print_objective_information(self, iterator, gradient_num):
        print("Iterator:", iterator)
        print("Objective value: ", self.objective_function(self.theta, self.length, self.point))
        print("Objective value first term: ",
              self.objective_function_first_term(0, self.theta, self.length, self.point))
        print("Objective value energy term: ", self.objective_function_energy_term(self.theta))
        print("Objective value penalty term: ", self.objective_function_penalty_term(self.theta))
        print("Mobius energy:",
              self.mobius_energy(func.calculate_entire_gamma_from_theta(self.theta, self.point, self.length)))
        print("Norm Gradient numerical: ", np.linalg.norm(gradient_num))

    def update_gamma(self):
        self.gamma = func.calculate_entire_gamma_from_theta(self.theta, self.point, self.length)
        self.gamma_der = func.calculate_entire_gamma_der_from_theta(self.theta, self.length)

    def objective_function(self, theta, length, point):
        obj = 0
        obj = self.objective_function_first_term(obj, theta, length, point)
        obj += self.objective_function_energy_term(theta)
        obj += self.objective_function_penalty_term(theta)
        return obj

    @staticmethod
    def quadratic_penalty_term(theta):
        cos_integral = func.trapezoidal_rule(np.cos(theta), 1, 0)
        sin_integral = func.trapezoidal_rule(np.sin(theta), 1, 0)
        return cos_integral ** 2 + sin_integral ** 2

    def objective_function_first_term(self, obj, theta, length, point):
        for angle, dictionary in self.angle_to_alphas_and_exact_radon.items():
            difference_in_radon = self.calc_radon_transform_minus_exact(dictionary[const.EXACT_RADON_TRANSFORM], angle,
                                                                        theta, length, point)
            difference_in_radon_squared = difference_in_radon ** 2
            l2_norm = func.trapezoidal_rule(difference_in_radon_squared, 1, 0)
            obj += l2_norm / 2.0
        return obj

    def objective_function_energy_term(self, theta):
        return self.beta / 2.0 * self.energy_function(theta)

    def objective_function_penalty_term(self, theta):
        return self.lamda * (OptimizationObjectBFGS.quadratic_penalty_term(theta))

    def energy_function(self, theta):
        der_theta = (theta[1:] - theta[:-1]) * const.N_TIME
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

    def calc_radon_transform_minus_exact(self, exact_transform, angle, theta, length, point):
        gamma = func.calculate_entire_gamma_from_theta(theta, point, length)
        gamma_der = func.calculate_entire_gamma_der_from_theta(theta, length)
        differences_from_exact = (
                func.radon_transform(gamma, gamma_der, angle, const.PIXELS) - exact_transform.T)
        return_vector = differences_from_exact[0]
        return return_vector

    def numerical_gradient_point(self, theta, length, point):
        former_obj = self.objective_function(theta, length, point)
        point[0] += const.EPSILON
        x_changed_obj = self.objective_function(theta, length, point)
        point[0] -= const.EPSILON
        point[1] += const.EPSILON
        y_changed_obj = self.objective_function(theta, length, point)
        der_p_x = (x_changed_obj - former_obj) / const.EPSILON
        der_p_y = (y_changed_obj - former_obj) / const.EPSILON
        return np.array([der_p_x, der_p_y])

    def numerical_gradient_theta(self, theta, length, point):
        derivatives = np.zeros(len(theta) - 1)
        former_obj = self.objective_function(theta, length, point)
        for i in range(const.N_TIME):
            derivatives[i] = self.numerical_derivative_theta_i(i, former_obj, deepcopy(theta), length, point)
        return derivatives

    def numerical_derivative_theta_i(self, i, former_obj, theta, length, point):
        coord = np.zeros(const.N_TIME + 1)
        if i == 0:
            coord[0] = const.EPSILON
            coord[-1] = const.EPSILON
        else:
            coord[i] = const.EPSILON
        theta += coord
        changed_obj = self.objective_function(theta, length, point)
        theta -= coord
        return (changed_obj - former_obj) / const.EPSILON

    def numerical_gradient_length(self, theta, length, point):
        former_obj = self.objective_function(theta, length, point)
        changed_obj = self.objective_function(theta, length + const.EPSILON, point)
        derivative = (changed_obj - former_obj) / const.EPSILON
        return derivative

    def geodesic_length_squared(self, gamma, i, j):
        index_difference = min(abs(i - j), (min(i, j) + len(gamma) - 1 - max(i, j)))
        return (index_difference * self.length / const.N_TIME) ** 2

    def mobius_energy(self, gamma):
        integrand = 0
        for i in range(len(gamma)):
            for j in range(len(gamma)):
                if i != j and abs((i - j)) != len(gamma) - 1:
                    integrand += 1 / OptimizationObjectBFGS.length_squared(gamma[i],
                                                                           gamma[j]) - 1 / self.geodesic_length_squared(
                        gamma, i, j)
        return integrand / const.N_TIME ** 2 * self.length ** 2

    @staticmethod
    def length_squared(point_1, point_2):
        return (point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2

    def reset_gamma(self, actual_gamma, actual_gamma_der):
        self.gamma = actual_gamma
        self.gamma_der = actual_gamma_der
