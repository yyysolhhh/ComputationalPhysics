'''
전산물리학_A3_이예솔하
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

#plt.rcParams['animation.ffmpeg_path'] = ''

'''h = float(input("높이 입력"))
v = float(input("속도 입력"))'''
h = 1
v = 1.5
cor_ground = 0.7  # 바닥 반발계수
cor_wall = 1  # 벽 반발계수

# 공간 그리기
fig = plt.figure(figsize=(14, 6))
ax = fig.add_subplot(111)
ax.set_xlim([-h, 2.5])
ax.set_ylim([-h, 0.5])

# 박스, 벽 그리기
box = plt.Rectangle((-h, -h), h, h, color="purple")
wall = plt.Rectangle((2, -h), 0.5, 0.5+h, color="black")
ax.add_patch(box)
ax.add_patch(wall)

# 초기값
x = 0
y = 0
vx = v
vy = 0

# 공 튕기기


def ball_update():
    global x, y, vx, vy, cor_ground, cor_wall

    if y <= -h:
        vy = -vy * cor_ground
        y = -h
    if x <= 0:
        vx = -vx * cor_wall
        x = 0
    elif x >= 2:
        vx = -vx * cor_wall
        x = 2

# 공 그리기


def update_plot(t, fig, scat):
    global x, y, vx, vy, v

    x = x + vx * 0.01
    vy = vy - 9.8 * 0.01
    y = y + vy * 0.01

    ball_update()
    scat.set_offsets([x, y])
    return scat,


scat = plt.scatter(x, y, s=300, c="red", edgecolors='black', marker='o')

anim = animation.FuncAnimation(
    fig, update_plot, fargs=(fig, scat), frames=1000, interval=5)
writer_mpeg = animation.FFMpegWriter(fps=60, extra_args=['-vcodec', 'libx264'])
writer_pillow = animation.PillowWriter(fps=20)

plt.show()
