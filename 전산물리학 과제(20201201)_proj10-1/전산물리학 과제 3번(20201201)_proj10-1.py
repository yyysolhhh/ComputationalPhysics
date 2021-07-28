"""
20201201_전산물리학 과제 3번_proj10-1_이예솔하
Animated 3D random walk
"""
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

# ffmpeg download site: https://ffmpeg.org/download.html#build-windows.
# It is recommanded that ffmpeg.exe is extracted in the directory as below
plt.rcParams['animation.ffmpeg_path'] = r'C:\\ffmpeg\\bin\\ffmpeg.exe'

# Fixing random state for reproducibility
np.random.seed(1)


def random_walk_lines(max_time=250, dims=2):
    """
    Create a line using a random walk algorithm
   """
    data_lines = []
    for i in range(n_particles):
        lineData = np.zeros((dims, max_time))
        lineData[:, 0] = 0.5  # np.random.rand(dims)
        for index in range(1, max_time):
            # scaling the random numbers by 0.1 so
            # movement is small compared to position.
            # subtraction by 0.5 is to change the range to [-0.5, 0.5]
            step = ((np.random.rand(dims) - 0.5) * 0.1)
            lineData[:, index] = lineData[:, index - 1] + step
            for i in range(0, 2):
                if lineData[i, index] < 0:
                    lineData[:, index] = lineData[:, index - 1] + np.abs(step)
                if lineData[i, index] > 1:
                    lineData[:, index] = lineData[:, index - 1] - np.abs(step)
        data_lines.append(lineData)
    return data_lines


def animate(i, dataLines, lines):  # update lines
    for line, data in zip(lines, dataLines):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(data[0:2, i-1:i])
        #line.set_3d_properties(data[2, i-1:i])
    return lines


n_particles = 500
max_time = 250
# Attaching 3D axis to the figure
fig = plt.figure()
ax = plt.axes()
#ax = p3.Axes3D(fig)

# Creating n_particles line objects.
data_lines = random_walk_lines()
lines = [ax.plot([], [], '.')[0] for j in range(n_particles)]

# Setting the axes properties
#ax.set_xlim3d([0.0, 1.0])
plt.xlim(0, 1.0)
plt.ylim(0, 1.0)
ax.set_xlabel('X')

#ax.set_ylim3d([0.0, 1.0])
ax.set_ylabel('Y')

#ax.set_zlim3d([0.0, 1.0])
# ax.set_zlabel('Z')

ax.set_title('3D random walk')

# Creating the Animation objectf
line_ani = animation.FuncAnimation(fig, animate, max_time, fargs=(data_lines, lines),
                                   interval=100, blit=False)
writer_mpeg = animation.FFMpegWriter(fps=60, extra_args=['-vcodec', 'libx264'])
writer_pillow = animation.PillowWriter(fps=20)

file_name = '3-D_random_walk_particle'
ext_mp4 = '.mp4'
ext_mov = '.mov'
ext_avi = '.avi'
ext_gif = '.gif'
#line_ani.save(file_name + ext_mp4, writer=writer_mpeg)
#line_ani.save(file_name + ext_mov, writer=writer_mpeg)
#line_ani.save(file_name + ext_avi, writer=writer_mpeg)
#line_ani.save(file_name + ext_gif, writer=writer_pillow)

plt.show()
