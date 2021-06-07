import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.stats import linregress

def integrand(x, epsilon):
    return 1 / (1 + np.exp(-2 * x / epsilon))


N = 1000
x_list = np.logspace(-10, -1, N)
big_L=100
small_L = 1
integral_list = np.zeros(N)
true_sol = np.zeros(N)
for i in range(N):
    small_L = 712*x_list[i]
    integral_list[i] = big_L*integrate.quad(integrand, 0, small_L, args=x_list[i])[0]
    true_sol[i] = big_L * small_L
plt.loglog(x_list, np.abs(true_sol - integral_list))
print(np.abs(true_sol - integral_list))
plt.show()

plt.plot(x_list, np.abs(true_sol - integral_list))
plt.show()


print(linregress(np.log(integral_list), np.log(true_sol)))
print(linregress(integral_list, np.log(true_sol)))
