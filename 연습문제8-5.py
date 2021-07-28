'''
전산물리학(20201210)_연습문제8-5_이예솔하
'''
import matplotlib.pyplot as plt
import numpy as np
import sympy
from scipy import optimize


def f(x, a0, a1):
    return (a1/2) / ((x - a0)**2 + (a1/2)**2)


xdata = np.array([10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5,
                  15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0])

# 실험 데이타
ydata = np.array([0.0216, 0.0279, 0.0367, 0.0523, 0.0763, 0.121, 0.239, 0.587, 1.96,
                  1.72, 0.503, 0.25, 0.121, 0.0723, 0.0498, 0.0351, 0.0278, 0.0216, 0.0162, 0.0145])


def g(p_a):
    return ydata - f(xdata, *p_a)


p_a_start = (1, 1)

p_a_opt, beta_cov = optimize.leastsq(g, p_a_start)
fig, ax = plt.subplots(figsize=(8, 6))

x = np.linspace(0, 30, 1001)
ax.scatter(xdata, ydata, label="samples")

ax.plot(x, f(x, *p_a_opt), 'b', lw=2, label="fitted model")
ax.set_xlim(10, 21)
ax.set_xlabel(r"$\hbar\omega$", fontsize=18)
ax.set_ylabel(r"$\sigma$", fontsize=18)
plt.text(0.1, 0.97, r'fitted parameters: $E_0$=%.5f, $\Gamma$=%.5f' %
         (p_a_opt[0], p_a_opt[1]), transform=ax.transAxes)
ax.legend()
plt.title('curve fitting using Levenberg-Marquardt method')
fig.tight_layout()
# fig.savefig('proj8-1_lesstsq.png')
plt.show()
