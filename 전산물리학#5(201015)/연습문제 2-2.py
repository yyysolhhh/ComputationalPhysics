'''
201015_전산물리학_연습문제2-2_이예솔하
'''

import numpy as np
from scipy import integrate


def f(x): return x**3 / (np.exp(x) - 1)


quad, error_quad = integrate.quad(f, 0, np.infty)
exact_value = np.pi**4 / 15

print("integration by quadrature= %.5e, error_quadrature= %.5e" %
      (quad, quad - exact_value))
print("exact_value= %.5e" % (exact_value))
