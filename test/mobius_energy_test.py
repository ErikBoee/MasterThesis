import numpy as np
import source.optimization_object_bfgs_utilities as opt_ut
from constants_test import N_TIME, PIXELS

radius = PIXELS / 3
init_length = 2 * np.pi * radius
point_ref = np.array([PIXELS / 2, PIXELS / 2 - radius])
length_ref = 2 * np.sin(np.pi / N_TIME) * N_TIME * radius

t_n = np.linspace(0, 1, N_TIME + 1)


def calc_theta_ref(tau):
    return 2 * np.pi * tau


def calc_gamma_ref():
    return opt_ut.calculate_entire_gamma_from_theta(calc_theta_ref(t_n), point_ref, length_ref)


theta_ref = calc_theta_ref(t_n)
init_theta = theta_ref
gamma_ref = calc_gamma_ref()


def length_squared(point_1, point_2):
    return ((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2) ** 4


def geodesic_length_squared(gamma, length, i, j, n):
    index_difference = min(abs(i - j), (min(i, j) + len(gamma) - 1 - max(i, j)))
    return (index_difference * length / n) ** 8


def interpolate(gamma, no_of_extra_points):
    new_gamma = np.zeros((((len(gamma) - 1) * (1 + no_of_extra_points) + 1), 2))
    index_multiplier = (1 + no_of_extra_points)
    for i in range(len(gamma) - 1):
        new_gamma[i * index_multiplier] = gamma[i]
        for j in range(no_of_extra_points):
            new_gamma[i * index_multiplier + j + 1] = (j + 1) / index_multiplier * gamma[i + 1] + (
                    no_of_extra_points - j) / index_multiplier * gamma[i]
    new_gamma[-1] = gamma[-1]
    opt_ut.draw_boundary(new_gamma, gamma_ref, 29, PIXELS)
    return new_gamma


def mobius_energy(length, gamma, n, obj_value):
    no_of_extra_points = get_number_of_extra_points(n - 1, length, obj_value)
    gamma = interpolate(gamma, no_of_extra_points)
    opt_ut.draw_boundary(gamma, gamma, 22, PIXELS)
    number_of_lengths = len(gamma) - 1
    len_gamma = len(gamma)
    print(number_of_lengths, n)
    integrand = 0
    for i in range(len_gamma):
        i_integrand = 0
        for j in range(len_gamma):
            if validation_check(i, j, no_of_extra_points, len_gamma):
                if j == 0 or j == len(gamma) - 1:
                    i_integrand += (1 / opt_ut.length_squared(gamma[i], gamma[j]) - 1 / geodesic_length_squared(
                        gamma, length, i, j,
                        number_of_lengths)) / 2
                else:
                    i_integrand += 1 / opt_ut.length_squared(gamma[i], gamma[j]) - 1 / geodesic_length_squared(
                        gamma, length, i, j,
                        number_of_lengths)
        if i == 0 or i == len(gamma) - 1:
            integrand += i_integrand / 2
        else:
            integrand += i_integrand
    return integrand / number_of_lengths ** 2 * length ** 2


def validation_check(i, j, no_of_extra_points, len_gamma):
    modulo = i % (1 + no_of_extra_points)
    if i < no_of_extra_points + 1:
        return 2 * (1 + no_of_extra_points) < j < len_gamma - 2 - no_of_extra_points

    elif i > len_gamma - 3 - no_of_extra_points:
        return no_of_extra_points + 1 < j < len_gamma - 1 - 2 * (1 + no_of_extra_points)

    else:
        return i + (no_of_extra_points + 1) - modulo + 1 + no_of_extra_points < j or j < i - modulo - (
                1 + no_of_extra_points)


def get_number_of_extra_points(len_gamma, length, obj_value):
    if obj_value == 1:
        return 0
    else:
        value = int(np.ceil((obj_value ** (1 / 8) * abs(length) + 1) / (np.sqrt(2) * len_gamma) - 1))
        return value


gamma_perfect_circle = np.zeros((N_TIME + 1, 2))
centrum = np.array([PIXELS / 2, PIXELS / 2])
i = 0
for angle in np.linspace(3 * np.pi / 2, 2 * np.pi + 3 * np.pi / 2, N_TIME + 1):
    gamma_perfect_circle[i] = np.add(centrum, np.array([radius * np.cos(angle), radius * np.sin(angle)]))
    i += 1
length_perfect_circle = 2 * radius * np.sin(np.pi / N_TIME) * N_TIME
#mobius_perfect_circle = mobius_energy(length_perfect_circle, gamma_perfect_circle, N_TIME, 1)
#mobius = mobius_energy(length_ref, gamma_ref, N_TIME, 1)
length_vector = opt_ut.get_length_vector(theta_ref, length_ref)
print(length_vector)
#mobius_2 = opt_ut.mobius_energy(gamma_ref, length_ref, N_TIME, 100, length_vector)
for j in range(3):
    for i in range(N_TIME * (j + 1)):
        path_length_not_crossing = opt_ut.get_not_crossing_start_path_length(0, i, 0, i % (j + 1), length_vector, j)
        path_length_crossing = opt_ut.get_crossing_start_path_length(0, i, 0, i % (j + 1), length_vector, j)
        #print("i:", i)
        #print("Crossing start: ", path_length_crossing)
        #print("Not crossing start: ", path_length_not_crossing)
#print(mobius_perfect_circle)
#print(mobius_2)
#print(mobius)
for j in range(16):
    for i in range(16):
        print(i, j, opt_ut.validation_check(i, j, 2, 16))
opt_ut.draw_boundary(gamma_ref, gamma_perfect_circle, -2, PIXELS)
