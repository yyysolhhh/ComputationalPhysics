'''
전산물리학(20201210)_연습문제8-3_이예솔하
'''
import matplotlib.pyplot as plt
import numpy as np
import sympy
from scipy import optimize

# Newton 법
x, y = sympy.symbols("x, y")
f = -sympy.exp(-3*x**2 - 2*x - x*y - 2*y**2 + 7*y)
fprime = [f.diff(x_) for x_ in (x, y)]  # Gradient
sympy.Matrix(fprime)
hessian = [[f.diff(x_, y_) for x_ in (x, y)] for y_ in (x, y)]  # Hessian

sympy.Matrix(hessian)
ff = sympy.lambdify((x, y), f, 'numpy')
ffprime = sympy.lambdify((x, y), fprime, 'numpy')
hhessian = sympy.lambdify((x, y), hessian, 'numpy')


def func(f):
    return lambda X: np.array(f(X[0], X[1]))


f = func(ff)
fprime = func(ffprime)
fhess = func(hhessian)
X_start = optimize.brute(
    f, (slice(-1, 0, 0.01), slice(1, 2, 0.01)), finish=None)
start_x, start_y = X_start[0], X_start[1]

# finding minimum
X_opt = optimize.fmin_ncg(f, (start_x, start_y), fprime=fprime, fhess=fhess)
f_max = f(X_opt)
print('Newton 법:', X_opt, -f_max)

# 최급경사탐색법


def f1(x):
    # 최솟값을 구할 함수
    return -np.exp(-(3*x[0]**2 + 2*x[0] + x[0]*x[1] + 2*x[1]**2 - 7*x[1]))


def dfdx(x):
    return -(-6*x[0] - 2 - x[1]) * np.exp(-(3*x[0]**2 + 2*x[0] + x[0]*x[1] + 2*x[1]**2 - 7*x[1]))


def dfdy(x):
    return -(-x[0] - 4*x[1] + 7) * np.exp(-(3*x[0]**2 + 2*x[0] + x[0]*x[1] + 2*x[1]**2 - 7*x[1]))


def gradient(x):
    return np.array([dfdx(x), dfdy(x)])


X_start = optimize.brute(
    f1, (slice(-1, 0, 0.01), slice(1, 2, 0.01)), finish=None)

# Learrning Rate
learning_rate = 0.01
tol = 1.e-8
current_X = X_start.copy()
for loop in range(100000):
    previous_X = -current_X.copy()
    current_X -= learning_rate*gradient(previous_X)
    if np.all(np.abs(current_X - previous_X) < tol):
        break
    #print("iteration count ", loop, current_X)
print('최급경사탐색법:', current_X, -f(current_X))

# Newton 법 그림
fig, ax = plt.subplots(figsize=(15, 5))
plt.subplot(121)
plt.xlim(-2, 0.7)
plt.ylim(0, 4)
x_ = np.linspace(-2, 0.7, 1000)
y_ = np.linspace(0, 3.7, 1000)
X, Y = np.meshgrid(x_, y_)
c = plt.contour(X, Y, -ff(X, Y), 100)
plt.plot(X_opt[0], X_opt[1], 'b.', markersize=15)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
plt.colorbar()
plt.text(X_opt[0]+0.5, X_opt[1]+1.5, 'max = %.7e \n at x=%.7e,\n y=%.7e' %
         (-f_max, X_opt[0], X_opt[1]), color='black')
plt.title('Finding maximum of multivariable function using Newton method')
fig.tight_layout()
# fig.savefig('ex8-3_newton.png')

# 최급경사탐색법 그림
plt.subplot(122)
plt.xlim(-2, 0.7)
plt.ylim(0, 4)
x1_ = np.linspace(-2, 0.7, 1000)
y1_ = np.linspace(0, 3.7, 1000)
X1, Y1 = np.meshgrid(x1_, y1_)
c1 = plt.contour(X1, Y1, -f1([X1, Y1]), 100)
plt.plot(current_X[0], current_X[1], 'b.', markersize=15)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
plt.colorbar()
plt.text(current_X[0]+0.5, current_X[1]+1.5, 'max = %.7e \n at x=%.7e, \n y=%.7e' %
         (-f(current_X), current_X[0], current_X[1]), color='black')
plt.title('Finding maximum of multivariable function using gradient descent method')
fig.tight_layout()
# fig.savefig('ex8-3_steepset_gradient_descent.png')

plt.show()
