import numpy as np
import matplotlib.pyplot as plt
import svgpathtools as svg
from source.functions import calculate_entire_gamma_from_theta
import cv2
import test_utilities as ut

# Import the curves from svg file
curves, attr = svg.svg2paths('star_shaped_test_2.svg')
c = curves[0]

print(c.length())  # computes the length of the curves
print(c.point(.3))  # computes point evaluations of the curves at time 0<=t<=1
print(c.normal(.3))  # computes the unit normal
print(c.unit_tangent(.3))  # computes the unit tangent
print(svg.path.inv_arclength(c, 30))  # estimates the inverse arc length at cumulative length 0<=s<=Length(c)

n = 100
t = [svg.path.inv_arclength(c, s) for s in np.linspace(0, c.length(), n + 1)]
print(t[-1])
points = np.array([c.point(t) for t in t])
tangents = np.array([c.unit_tangent(t) for t in t])

plt.quiver(points.real, -points.imag, tangents.real, -tangents.imag)
plt.show()

plt.plot(points.real, points.imag)
plt.show()

theta = np.angle(tangents)
theta[-1] = theta[0] + 2*np.pi

pixels = 500
radius = pixels / 3
point = (pixels / 2 - radius/2, pixels / 2)
length = np.pi * radius
gamma = calculate_entire_gamma_from_theta(theta, point, length)
boundary_image = np.zeros((pixels, pixels, 3), np.uint8)

for gamma_value in gamma:
    boundary_image[int(gamma_value[0]), int(gamma_value[1])] = [255, 255, 255]

cv2.imwrite(str(pixels) + "_x_" + str(pixels) + "_boundary.png", boundary_image)
