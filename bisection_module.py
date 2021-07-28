'''
전산물리학_bisection module_이예솔하
'''


def bisection(func, *args, numberOfRoot=1, xL=-10, h=0.1, tol=1.e-8):
    import numpy as np
    sol_x = []
    x = xL
    for i in range(numberOfRoot):
        loop = 0
        hx = h

        if i > 0:
            x = sol_x[i-1] + hx

        while 1:
            loop += 1
            #print("loop=%d, x=%e, f(x)=%e" % (loop, x, func(x)))
            x += hx
            if loop > 100000:
                x = 50000
                print("error: too many loops!")
                break
            if(np.fabs(func(*args, x)) < tol):
                break
            if(func(*args, x) * func(*args, x-hx) < 0):
                x -= hx
                hx *= 0.5
        sol_x.append(x)
    return sol_x
