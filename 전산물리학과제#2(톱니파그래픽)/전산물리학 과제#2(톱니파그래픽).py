# 전산물리학 과제2(톱니파 그래픽)_이예솔하

import numpy as np
from matplotlib import pyplot as plt

plt.figure(figsize=(8, 3), dpi=80)  # 8 * 3인치 그래프 생성

# t를 -2pi와 2pi 사이에서 200개의 점을 원소로 하는 배열로 정의
t = np.linspace(-2 * np.pi, 2 * np.pi, 200)
cosine, sine = np.cos(t), np.sin(t)  # t를 변수로 하는 cosine, sine 함수값이 원소인 배열 정의
w = 1

f1, f2, f3 = 0, 0, 0

for n in range(1, 6):
    f1 = f1 + ((-1) * np.sin(n * w * t) / (n * np.pi))

for n in range(1, 11):
    f2 = f2 + ((-1) * np.sin(n * w * t) / (n * np.pi))

for n in range(1, 16):
    f3 = f3 + ((-1) * np.sin(n * w * t) / (n * np.pi))


plt.subplot(1, 3, 1)  # f1 그래프 그리기
plt.plot(t, f1, color="red", linewidth=1.0, linestyle="-")
plt.legend(["n==5"], bbox_to_anchor=(0.5, 1.1))
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

plt.subplot(1, 3, 2)  # f2 그래프 그리기
plt.plot(t, f2, color="blue", linewidth=1.0, linestyle="-")
plt.legend(["n==10"], bbox_to_anchor=(0.5, 1.1))
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

plt.subplot(1, 3, 3)  # f3 그래프 그리기
plt.plot(t, f3, color="purple", linewidth=1.0, linestyle="-")
plt.legend(["n==15"], bbox_to_anchor=(0.5, 1.1))
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

# plt.savefig("이예솔하_전산물리학과제#2(톱니파그래픽)")
plt.show()
