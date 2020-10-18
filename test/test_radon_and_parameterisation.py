import numpy as np
import cv2
import matplotlib.pyplot as plt
from copy import deepcopy
from skimage.transform import radon
import source.functions as func
import test_utilities as ut

pixels = 500
radius = pixels / 3
point = (pixels / 2, pixels / 2 - radius)
length = 2 * np.pi * radius
n = 300


def draw_boundary_and_filled_image(entire_gamma, t_n, pixels):
    boundary_image = np.zeros((pixels, pixels, 3), np.uint8)
    filled_image_radon = ut.create_image_from_curve(entire_gamma, pixels, t_n)
    filled_image = np.zeros((pixels, pixels, 3))
    filled_image[filled_image_radon == 1.0] = [255, 255, 255]

    for gamma_value in entire_gamma:
        boundary_image[int(gamma_value[0]), int(gamma_value[1])] = [255, 255, 255]

    cv2.imwrite(str(pixels) + "_x_" + str(pixels) + "_boundary.png", boundary_image)
    cv2.imwrite(str(pixels) + "_x_" + str(pixels) + "_filled.png", filled_image)


t_n = np.linspace(0, 1, n)
entire_gamma = func.calculate_entire_gamma(t_n, point, length)
entire_der_gamma = func.calculate_entire_gamma_der(t_n, length)

angle_list = np.linspace(0, np.pi, 10)

filled_radon_image = ut.create_image_from_curve(entire_gamma, pixels, t_n)
for angle in angle_list:
    radon_trans_own = func.radon_transform(entire_gamma, entire_der_gamma, angle, pixels)
    radon_transform_py = radon(filled_radon_image, theta=[ut.rad_to_deg(angle)], circle=True)
    plt.title("Angle = " + str(ut.rad_to_deg(angle)))
    plt.plot(np.linspace(0, pixels, pixels), radon_trans_own, label="Radon calculated")
    plt.plot(np.linspace(0, pixels, pixels), radon_transform_py, label="Radon built in")
    plt.legend()
    plt.show()

draw_boundary_and_filled_image(entire_gamma, t_n, pixels)
