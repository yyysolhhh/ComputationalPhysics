'''
20201105_전산물리학_double pendulum_이예솔하
'''

import matplotlib.animation as animation
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

L1 = 1.0
L2 = 1.0
M1 = 1.0
M2 = 1.0
g = 9.8


def derivs(y0, x):
    """ differential equations"""
    delta_theta = y0[0] - y0[2]

    dydx = np.zeros_like(y0)

    alpha1 = np.cos(delta_theta) * L2 * M2 / (L1 * (M1 + M2))
    alpha2 = np.cos(delta_theta) * L1 / L2
    f1 = -L2 * M2 / (L1 * (M1 + M2)) * \
        (y0[3])**2 * np.sin(delta_theta) - g * np.sin(y0[0]) / L1
    f2 = L1 / L2 * (y0[1])**2 * np.sin(delta_theta) - g * np.sin(y0[2]) / L2
    dydx[0] = y0[1]
    dydx[1] = (f1 - alpha1 * f2) / (1 - alpha1 * alpha2)
    dydx[2] = y0[3]
    dydx[3] = (-alpha2 * f1 + f2) / (1 - alpha1 * alpha2)
    return dydx


n_steps = 10000
x0 = 0
xn = 500
x = np.linspace(x0, xn, n_steps+1)
dx = (xn - x0)/n_steps

# initial state(degrees)
theta1 = 90.
dtheta1 = 0.
theta2 = 90.
dtheta2 = 0.

y0 = np.radians([theta1, dtheta1, theta2, dtheta2])

# integrate your ODE using scipy.integrate.
y = integrate.odeint(derivs, y0, x)

x1 = L1 * np.sin(y[:, 0])
y1 = -L1 * np.cos(y[:, 0])
x2 = x1 + L2 * np.sin(y[:, 2])
y2 = y1 - L2 * np.cos(y[:, 2])


fig = plt.figure()
ax = plt.subplot(111)
line, = ax.plot([], [], 'o-', lw=2)
plt.xlim(-3, 3)
plt.ylim(-3, 3)
ax.grid()

time_template = 'time = %.1f'
time_text = ax.text(0.1, 0.9, '', transform=ax.transAxes)


def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


def animate(i):
    line.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
    time_text.set_text(time_template % (i*dx))
    return line, time_text


ani = animation.FuncAnimation(fig, animate, range(1, len(y)),
                              interval=dx, blit=True, init_func=init)

plt.show()
