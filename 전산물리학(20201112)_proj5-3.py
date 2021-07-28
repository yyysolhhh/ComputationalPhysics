'''
전산물리학(20201112)_proj5-3_이예솔하
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


def rk4(func, y0, x):
    y0 = np.array(y0)
    n = len(x)
    def f(xi, yi): return np.array(func(yi, xi))
    y = np.zeros((n,) + y0.shape)
    h = (x[n-1] - x[0])/(n-1)
    xi = x[0]
    y[0] = yi = y0[:]
    for i in range(1, n):
        k1 = h * f(xi, yi)
        k2 = h * f(xi + 0.5 * h, yi + 0.5 * k1)
        k3 = h * f(xi + 0.5 * h, yi + 0.5 * k2)
        k4 = h * f(xi + h, yi + k3)
        xi = x[i]
        y[i] = yi = yi + (k1 + 2*(k2 + k3) + k4) / 6
    return y


def bisect(z, del_z, zc):
    global lp, lm, z_minus, z_plus
    if zc > 0.:
        lp = 1
        z_plus = z
    else:
        lm = 1
        z_minus = z
    if lp*lm == 0:
        z += (lm - lp)*del_z
        return(z)
    else:
        z = 0.5*(z_plus + z_minus)
        return z


def potential(x):
    return 0.5*x*x + c4*x**4


def derivs(y0, x):
    dydx = np.zeros_like(y0)
    dydx[0] = y0[1]
    dydx[1] = 2.*(potential(x)-E)*y0[0]
    return dydx


n_steps = 10000
tol = 1.e-5
x0 = -3  # x=0
xn = -x0  # x=1m

E_i = 0.000001
a_asymp = 10.
del_E = 0.1
nth_max = 5  # = input('input number of energy levels')

yn = 0  # An exact value of y at right point.
# create a x array from x0 to xn.
x = np.linspace(x0, xn, n_steps+1)
c4 = 0.1

E = E_i
E_all = []

for nth in range(nth_max):
    del_k = (-1)**(nth+1) * 0.5

    # global variable set for function bisect()
    loop = lp = lm = 0
    z_plus = z_ninus = 0.
    yc = 100000
    E_old = -100000

    while np.fabs(yc) > tol and np.fabs(E_old - E) > 1.e-100:
        y00 = a_asymp * E * E * np.exp(1/3*np.sqrt(2*c4)*x0*x0*x0)
        y01 = np.sqrt(2*c4)*x0**2 * y00
        y0 = [y00, y01]
        y = rk4(derivs, y0, x)
        yc = ((-1)**nth)*(((-1)**nth)*y00 - y[-1, 0])/y00
        E_old = E
        E = bisect(E, del_E, yc)
        loop += 1
        print('%dth bound state, loop = %d,  E = %.14f pi, yc = %.5e' %
              (nth, loop, E, yc))
        if loop > 100000:
            break
#    input()
    print('\n')
    E_all.append(E)  # append all k's to k_all
    E = E + np.fabs(del_E)

fig = plt.figure(figsize=(12, 8))
ax = plt.gca()
plt.xlim(x0, -x0)
plt.ylim(-1.5, 1.5)
plt.title(
    r'Wave functions of time-independent Schr$\..{o}$dinger equation for a harmonic osc. case.')

for iE in range(len(E_all)):
    E = E_all[iE]
    y = rk4(derivs, y0, x)
    y[:, 0] /= np.sqrt(integrate.simps(y[:, 0]*y[:, 0], x))
    plt.text(0.7, 0.95-iE*0.04, r'%dth bound state energy = %.5f $\hbar \omega$' %
             (iE, E), transform=ax.transAxes)
    line, = ax.plot(x, y[:, 0], lw=0.8)
    ax.plot(x, potential(x), 'b', lw=5, alpha=0.05)

plt.grid()
# plt.savefig('proj5-2_harmonic_osc.png')
plt.show()
