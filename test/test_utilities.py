import numpy as np
import numba as nb
from numba import njit


def rad_to_deg(rad):
    return 180 * rad / np.pi


@njit(fastmath=True)
def calculate_winding_number(point_to_wind, entire_gamma):
    gamma_current_subtracted = np.subtract(entire_gamma[:-1], point_to_wind)
    gamma_next_subtracted = np.subtract(entire_gamma[1:], point_to_wind)
    gamma_next_angle = np.arctan2(gamma_next_subtracted[:, 1], gamma_next_subtracted[:, 0])
    gamma_current_angle = np.arctan2(gamma_current_subtracted[:, 1], gamma_current_subtracted[:, 0])
    signed_angles = np.subtract(gamma_next_angle, gamma_current_angle)
    signed_angles[signed_angles > np.pi] -= 2 * np.pi
    signed_angles[signed_angles < -np.pi] += 2 * np.pi
    winding_number = np.sum(signed_angles)
    return round(winding_number / (2 * np.pi))


def signed_angle_out_of_scope(signed_angle):
    return abs(signed_angle) > np.pi


def update_signed_angle(signed_angle):
    if signed_angle > np.pi:
        return signed_angle - 2 * np.pi
    return signed_angle + 2 * np.pi


def get_current_and_next_vector(entire_gamma, i, point_to_wind):
    current_gamma = entire_gamma[i]
    next_gamma = entire_gamma[i + 1]
    return current_gamma - point_to_wind, next_gamma - point_to_wind


@njit
def create_image_from_curve(entire_gamma, pixels, t_list):
    img = np.zeros((pixels, pixels), nb.float_)
    boundary_pixels = set()
    number_of_times = len(t_list)
    for i in range(number_of_times):
        x, y = entire_gamma[i]
        img[round(x), round(y)] = 1.0
        boundary_pixels.add((round(x), round(y)))
    for i in range(pixels):
        for j in range(pixels):
            if not (i, j) in boundary_pixels and calculate_winding_number(np.array([i, j]), entire_gamma) == 1:
                img[i, j] = 1.0
    return img
