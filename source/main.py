from source.initializer import beta, theta_ref, angle_to_exact_radon,\
    gamma_ref, init_point, init_length, init_theta
import source.optimization_object as opt

if __name__ == '__main__':
    opt_object = opt.QuadraticPenalty(init_theta, init_length, init_point, theta_ref,
                                      gamma_ref, angle_to_exact_radon, beta)
    opt_object.gradient_descent()


