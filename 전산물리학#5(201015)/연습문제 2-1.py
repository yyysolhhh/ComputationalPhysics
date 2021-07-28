'''
201015_전산물리학_연습문제2-1_이예솔하
'''

import numpy as np
from scipy import integrate


def f(x): return 2 * x**2 + x**4


x_down, x_up, steps = 0, 2, 1000
x = np.linspace(x_down, x_up, steps+1)
y = f(x)
exact_value = 176 / 15

steps = [20, 200, 2000]
for i in steps:
    x_down, x_up = 0, 2
    x = np.linspace(x_down, x_up, i+1)
    y = f(x)
    exact_value = 176 / 15

    trapezoid = integrate.trapz(y, x)
    simpson = integrate.simps(y, x)
    quad, error_quad = integrate.quad(f, 0, 2)

    print("h == %f" % (2/i))
    print("integration by trapezoid == %.5e, error_trapezoid = %.5e" %
          (trapezoid, trapezoid - exact_value))
    print("integration by simpson == %.5e, error_simpson = %.5e" %
          (simpson, simpson - exact_value))
    #print("integration by quadrature == %.5e, error_quadrature= %.5e" %(quad, error_quad))
print("integration by quadrature == %.5e, error_quadrature= %.5e" %
      (quad, error_quad))

# Monte Carlo


def monte(f, x_left, x_right, n_throw):
    mc_x = np.random.random(n_throw) * (x_right - x_left) + x_left
    y_sum = f(mc_x).sum()
    mc_y = y_sum * (x_right - x_left)/n_throw
    return mc_y


n = 7
montegral = monte(f, 0, 2, 10**n)
error = (montegral - exact_value)/exact_value
print("no. of throws == 10^%d, integration by montecarlo == %.5e, error_monte == %.5e" %
      (n, montegral, error))
