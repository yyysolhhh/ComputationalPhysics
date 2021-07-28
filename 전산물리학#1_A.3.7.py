'''
전산물리학_A.3.7_이예솔하
'''
w = 1

for i in range(1, 1000001):
    w = w * 4 * i * i/(4 * i * i - 1)

pi = 2 * w
print(pi)
