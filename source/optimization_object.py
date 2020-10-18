import numpy as np


class QuadraticPenaltyOptimizationCurve():
    def __init__(self, init_theta, init_length, init_point, theta_ref, exact_radon_transforms):
        self.theta = init_theta
        self.length = init_length
        self.point = init_point
        self.theta_ref = theta_ref
        self.exact_radon_transforms = exact_radon_transforms




