'''
전산물리학(20201126)_연습문제7-1_이예솔하
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def initial_con(x):
    dydt = np.zeros_like(x)
    #dydt[:] = 0
    n = 3000.
    y0 = n / (np.pi * (1.+n**2*(x - 0.5)**2)) - n / (np.pi * (1.+0.5**2*n**2))
    return dydt, y0


def PDE_solver(loop, x, dydt):
    nxp1 = len(x)
    if loop == 1:
        for i in range(1, nxp1-1):
            y[1, i] = 0.5 * (y[0, i+1] + y[0, i-1]) + ht*dydt[i]
    else:
        for i in range(1, nxp1-1):
            y[loop, i] = y[loop-1, i+1] + y[loop-1, i-1] - y[loop-2, i]


nx = 1000
nt = 10000
xL, xR = 0, 1
ht = 0.1
x = np.linspace(xL, xR, nx+1)
y = np.zeros((nt+1, nx+1))
dydt, y[0] = initial_con(x)

for it in range(1, nt+1):
    PDE_solver(it, x, dydt)


fig = plt.figure(figsize=(8, 6))
plt.title('animation of a wave on a string - PDE')
ax = plt.gca()
plt.xlim(xL, xR)
plt.ylim(-1.5, 1.5)
line, = plt.plot([], [], 'r-', lw=0.5)
ax.grid()
ax.set_ylabel(r"$y(t)$", rotation='horizontal')
ax.set_xlabel(r"$x$")

time_template = 'time = %.1f'
time_text = ax.text(0.43, 0.95, '', transform=ax.transAxes)


def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text,


def animate(i):

    line.set_data(x, y[i])
    time_text.set_text(time_template % (i*ht))
    return line, time_text,


ani = animation.FuncAnimation(fig, animate, range(1, len(y)),
                              interval=ht, blit=True, init_func=init, repeat=False)

plt.show()
