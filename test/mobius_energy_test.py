import numpy as np
import functions as func
from constants import N_TIME, PIXELS

radius = PIXELS / 3
init_length = 2 * np.pi * radius
point_ref = np.array([PIXELS / 2, PIXELS / 2 - radius])
length_ref = 2 * np.sin(np.pi / N_TIME) * N_TIME * radius

t_n = np.linspace(0, 1, N_TIME + 1)


def calc_theta_ref(tau):
    return 2 * np.pi * tau


def calc_gamma_ref():
    return func.calculate_entire_gamma_from_theta(calc_theta_ref(t_n), point_ref, length_ref)


theta_ref = calc_theta_ref(t_n)
init_theta = theta_ref
gamma_ref = calc_gamma_ref()


def length_squared(point_1, point_2):
    return (point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2


def geodesic_length_squared(length, gamma, i, j):
    index_difference = min(abs(i - j), (min(i, j) + len(gamma) - 1 - max(i, j)))
    return (index_difference * length / N_TIME) ** 2


def mobius_energy(length, gamma):
    integrand = 0
    for i in range(len(gamma)):
        i_integrand = 0
        for j in range(len(gamma)):
            if i != j and abs((i - j)) != len(gamma) - 1:
                if j == 0 or j == len(gamma) - 1:
                    i_integrand += (1 / length_squared(gamma[i], gamma[j]) - 1 / geodesic_length_squared(length, gamma,
                                                                                                         i, j)) / 2
                else:
                    i_integrand += 1 / length_squared(gamma[i], gamma[j]) - 1 / geodesic_length_squared(length, gamma,
                                                                                                        i, j)
        if i == 0 or i == len(gamma) - 1:
            integrand += i_integrand / 2
        else:
            integrand += i_integrand
    return integrand / N_TIME ** 2 * length ** 2


gamma_perfect_circle = np.zeros((N_TIME + 1, 2))
centrum = np.array([PIXELS / 2, PIXELS / 2])
i = 0
for angle in np.linspace(3*np.pi/2, 2 * np.pi + 3*np.pi/2, N_TIME + 1):
    gamma_perfect_circle[i] = np.add(centrum, np.array([radius * np.cos(angle), radius * np.sin(angle)]))
    i += 1
length_perfect_circle = 2 * radius * np.sin(np.pi / N_TIME) * N_TIME
print(length_perfect_circle, length_ref)
mobius_perfect_circle = mobius_energy(length_perfect_circle, gamma_perfect_circle)
mobius = mobius_energy(length_ref, gamma_ref)
print(mobius_perfect_circle)
print(mobius)
print(gamma_perfect_circle)
print(gamma_ref)
func.draw_boundary(gamma_ref, gamma_perfect_circle, -2, PIXELS)
