import numpy as np
from constants import N_TIME, PIXELS
import svgpathtools as svg

problem_name = 'Even_better_attempt_bump_prob_3'
# Import the curves from svg file
curves, _ = svg.svg2paths('Svg_drawings/' + problem_name +'/star_shaped_reference.svg')
c_ref = curves[0]

curves, _ = svg.svg2paths('Svg_drawings/' + problem_name + '/star_shaped_test_2.svg')
c_sol = curves[0]
n = N_TIME
t_ref = np.array([svg.path.inv_arclength(c_ref, s) for s in np.linspace(0, c_ref.length(), n + 1)])
t_sol = np.array([svg.path.inv_arclength(c_sol, s) for s in np.linspace(0, c_sol.length(), n + 1)])
tangents_ref = np.array([c_ref.unit_tangent(t) for t in t_ref])
tangents_sol = np.array([c_sol.unit_tangent(t) for t in t_sol])

theta_ref = np.angle(tangents_ref)
theta_sol = np.angle(tangents_sol)
theta_ref[-1] = theta_ref[0] + 2 * np.pi
theta_sol[-1] = theta_sol[0] + 2 * np.pi

radius = PIXELS / 3
init_point = np.array([PIXELS / 2 + 5, PIXELS / 2 + 5])
init_length = np.pi * radius
point_ref = np.array([PIXELS / 2, PIXELS / 2])
length_ref = np.pi * radius
point_sol = point_ref
length_sol = length_ref
init_theta = theta_ref
