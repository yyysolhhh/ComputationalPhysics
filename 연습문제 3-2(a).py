'''
201022_전산물리학_연습문제3-2_이예솔하
'''
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate


def rk4(func, y0, x):
    y0 = np.array(y0)
    n = len(x)
    def f(xi, yi): return np.array(func(yi, xi))
    y = np.zeros((n,) + y0.shape)
    h = (x[n-1] - x[0]) / (n-1)
    xi = x[0]
    y[0] = yi = y0[:]
    for i in range(1, n):
        k1 = h * f(xi, yi)
        k2 = h * f(xi + 0.5 * h, yi + 0.5 * k1)
        k3 = h * f(xi + 0.5 * h, yi + 0.5 * k2)
        k4 = h * f(xi + h, yi + k3)
        xi = x[i]
        y[i] = yi = yi + (k1 + 2 * (k2 + k3) + k4) / 6
    return y


def derives(y0, x):
    dydx = np.zeros_like(y0)
    dydx[0] = y0[1]
    dydx[1] = (2 * x * y0[1] - n * (n + 1) * y0[0]) / (1 - x**2)
    return dydx


def exact_y(x):
    return (35 * x**4 - 30 * x**2 + 3) / 8


n_steps = 1000
x0 = 2
xn = 5
n = 4

# create a x array from x0 to xn.
x = np.linspace(x0, xn, n_steps+1)

# initial state
y0 = [443/8, 125]

# integrate your ODE using scipy.integrate.
y_int = integrate.odeint(derives, y0, x)
y_rk4 = rk4(derives, y0, x)

#f = open("ex3-2_%d.txt" %n_steps, "w")
for i in range(n_steps+1):
    #print("x=%.4f y_exact=%.4e, error_odeint=%.4e, error_rk4=%.5e" %(x[i], exact_y(x[i]), y_int[i, 0]-exact_y(x[i]), y_rk4[i, 0]-exact_y(x[i])), file=f)
    print("x=%.4f y_exact=%.4e, error_odeint=%.4e, error_rk4=%.5e" % (
        x[i], exact_y(x[i]), y_int[i, 0]-exact_y(x[i]), y_rk4[i, 0]-exact_y(x[i])))
# f.close()
