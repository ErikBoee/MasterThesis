import numpy as np
from copy import deepcopy
from source.initializer import beta, theta_ref, angle_to_exact_radon, \
    gamma_ref, init_point, init_length, init_theta, theta_solution
import source.derivatives as der
import source.optimization_object as opt
import source.constants as const
import source.functions as func

opt_object = opt.QuadraticPenalty(init_theta, init_length, init_point, theta_solution,
                                  gamma_ref, angle_to_exact_radon, beta)


def numerical_gradient_theta_energy_func(opt_object):
    derivatives = np.zeros(const.N_TIME)
    actual_theta = deepcopy(opt_object.theta)
    former_obj = opt_object.energy_function()
    for i in range(const.N_TIME):
        coord = np.zeros(const.N_TIME + 1)
        if i == 0:
            coord[0] = const.EPSILON
            coord[-1] = const.EPSILON
        else:
            coord[i] = const.EPSILON
        opt_object.theta += coord
        changed_obj = opt_object.energy_function()
        derivatives[i] = (changed_obj - former_obj) / const.EPSILON
        opt_object.theta = deepcopy(actual_theta)
    return derivatives


def numerical_gradient_theta_radon(opt_object, angle):
    derivatives = np.zeros(const.N_TIME)
    actual_theta = deepcopy(opt_object.theta)
    gamma_vector = func.calculate_entire_gamma_from_theta(opt_object.theta, opt_object.point, opt_object.length)
    gamma_der_vector = func.calculate_entire_gamma_der_from_theta(opt_object.theta, opt_object.length)
    former_obj = sum(func.radon_transform(gamma_vector, gamma_der_vector, angle, const.PIXELS))
    for i in range(const.N_TIME):
        coord = np.zeros(const.N_TIME + 1)
        if i == 0:
            coord[0] = const.EPSILON
            coord[-1] = const.EPSILON
        else:
            coord[i] = const.EPSILON
        opt_object.theta += coord
        gamma_vector = func.calculate_entire_gamma_from_theta(opt_object.theta, opt_object.point, opt_object.length)
        gamma_der_vector = func.calculate_entire_gamma_der_from_theta(opt_object.theta, opt_object.length)
        changed_obj = sum(func.radon_transform(gamma_vector, gamma_der_vector, angle, const.PIXELS))
        derivatives[i] = (changed_obj - former_obj) / const.EPSILON
        opt_object.theta = deepcopy(actual_theta)
    return derivatives


def numerical_gradient_theta_quadratic_penalty(opt_object):
    derivatives = np.zeros(const.N_TIME)
    actual_theta = deepcopy(opt_object.theta)
    former_obj = opt_object.lamda * opt_object.quadratic_penalty_term()
    for i in range(const.N_TIME):
        coord = np.zeros(const.N_TIME + 1)
        if i == 0:
            coord[0] = const.EPSILON
            coord[-1] = const.EPSILON
        else:
            coord[i] = const.EPSILON
        opt_object.theta += coord
        changed_obj = opt_object.lamda * opt_object.quadratic_penalty_term()
        derivatives[i] = (changed_obj - former_obj) / const.EPSILON
        opt_object.theta = deepcopy(actual_theta)
    return derivatives


def test_length_derivative(opt_object):
    num_grad_length = opt_object.numerical_gradient_length()
    grad_length = opt_object.gradient_length()
    print("Length_grad norm error: ", np.linalg.norm(num_grad_length - grad_length))


def test_theta_derivative(opt_object):
    num_grad_theta_energy, grad_theta_energy = theta_energy_gradients(opt_object)
    print("Theta_energy_gradient norm error: ", np.linalg.norm(num_grad_theta_energy - grad_theta_energy))
    num_grad_theta_radon, grad_theta_radon = theta_radon_gradients(opt_object)
    print("Theta_radon_gradient norm error: ", np.linalg.norm(num_grad_theta_radon - grad_theta_radon))
    print(num_grad_theta_radon, grad_theta_radon)
    print("Theta_full_gradient norm error: ",
          np.linalg.norm(opt_object.numerical_gradient_theta() - opt_object.gradient_theta()))
    print(opt_object.numerical_gradient_theta(), opt_object.gradient_theta())
    num_grad_theta_QP, grad_theta_QP = theta_QP_gradients(opt_object)
    print("Theta_QP_gradient norm error: ",
          np.linalg.norm(num_grad_theta_QP - grad_theta_QP))


def theta_QP_gradients(opt_object):
    opt_object.theta = np.linspace(0, 1, const.N_TIME + 1)
    opt_object.theta[-1] = 2 * np.pi
    num_grad_theta_quadratic_penalty = numerical_gradient_theta_quadratic_penalty(opt_object)
    grad_theta_quadratic_penalty = opt_object.der_quadratic_penalty_term_diff_theta()
    return num_grad_theta_quadratic_penalty, grad_theta_quadratic_penalty


def theta_radon_gradients(opt_object):
    angle = 3 * np.pi / 4
    numerical_radon_gradient = numerical_gradient_theta_radon(opt_object, angle)
    radon_gradient = np.zeros(const.N_TIME)
    alphas = func.get_alphas(angle, const.PIXELS)
    basis_vector = np.array([np.cos(angle), np.sin(angle)])
    basis_vector_orthogonal = np.array([-basis_vector[1], basis_vector[0]])
    for alpha in alphas:
        gamma_diff_theta = func.gamma_diff_theta(opt_object.theta, opt_object.length)
        der_gamma_diff_theta = func.der_gamma_diff_theta(opt_object.theta, opt_object.length)
        radon_gradient += der.der_radon_diff_theta_given_alpha(alpha, basis_vector, basis_vector_orthogonal,
                                                               opt_object.gamma, opt_object.gamma_der, gamma_diff_theta,
                                                               der_gamma_diff_theta)
    return numerical_radon_gradient, radon_gradient


def theta_energy_gradients(opt_object):
    num_grad_theta_energy = numerical_gradient_theta_energy_func(opt_object)
    grad_theta_energy = opt_object.der_energy_func_theta()
    return num_grad_theta_energy, grad_theta_energy


def test_gamma_diff_theta():
    theta = np.linspace(0, 2 * np.pi, const.N_TIME + 1)
    length = 1
    print(func.gamma_diff_theta(theta, length))


def test_point_derivative(opt_object):
    opt_object.point -= np.array([0.3, 0])
    num_grad_point = opt_object.numerical_gradient_point()
    grad_point = opt_object.gradient_point()
    print("Point_gradient norm error: ", np.linalg.norm(grad_point - num_grad_point), num_grad_point, grad_point)
    opt_object.gamma = func.calculate_entire_gamma_from_theta(opt_object.theta, opt_object.point, opt_object.length)


#test_gamma_diff_theta()
test_length_derivative(opt_object)
test_point_derivative(opt_object)
test_theta_derivative(opt_object)
