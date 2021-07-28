'''
전산물리학_연습문제1-4_이예솔하
'''
import numpy as np
from bisection_module import bisection

g = 9.8
v0 = 50
b = 0.5
theta = [20, 30, 40, 50, 60]


def f(t):
    return (-g * t / b + (1 - np.exp(-b * t)) * (b * v0 * np.sin(np.deg2rad(i)) + g) / b**2)


for i in theta:
    t = bisection(f, xL=0.01, numberOfRoot=1)
    print("theta == %i일 때 걸리는 시간:" % i, t)
