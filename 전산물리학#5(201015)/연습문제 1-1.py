'''
201015_전산물리학_연습문제1-1_이예솔하
'''

import numpy as np


def func(x):
    return (252 * x**4 - 468 * x**3 - 415 * x**2 + 21 * x + 10)


def dfunc(x):
    return(1008 * x**3 - 1404 * x**2 - 830 * x + 21)


def newton_raphson():
    no_of_roots = 4
    tol = 1.e-8
    sol_x = []
    xL = [-5, -0.1, 1, 2]
    for i in range(no_of_roots):
        x = xL[i]
        loop = 0
        while 1:
            loop += 1
            if loop > 5000:
                sol_x.append(50000)
                break
            x = x - func(x)/dfunc(x)
            print("loop=%d, x=%d, f(x)=%e" % (loop, x, func(x)))
            if np.fabs(func(x)) < tol:
                break
        sol_x.append(x)
    return sol_x


print("roots:", newton_raphson())
