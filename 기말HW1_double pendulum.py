"""
전산물리학_기말HW1_double pendulum_이예솔하_2016039616
(proj3-2_ani(1) + 전산물리학(20201105)_double pendulum_2016039616_이예솔하 + prof6-1_fft_damped_DP.py)
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy import fftpack
import matplotlib.animation as animation

def derivs(y0, x):
    """ differential equations"""
    delta_theta = y0[0] - y0[2]

    dydx = np.zeros_like(y0)

    alpha1 = np.cos(delta_theta) * L2 * M2 / (L1 * (M1 + M2))
    alpha2 = np.cos(delta_theta) * L1 / L2
    f1 = -L2 * M2 / (L1 * (M1 + M2)) * (y0[3])**2 * np.sin(delta_theta) - g * np.sin(y0[0])/ L1
    f2 = L1 / L2 * (y0[1])**2 * np.sin(delta_theta) - g * np.sin(y0[2]) / L2
    dydx[0] = y0[1]
    dydx[1] = (f1 - alpha1 * f2) / (1 - alpha1 * alpha2)
    dydx[2] = y0[3]
    dydx[3] = (-alpha2 * f1 + f2) / (1 - alpha1 * alpha2)
    return dydx

L1 = 1.0
L2 = 1.0
M1 = 1.0
M2 = 1.0
g = 9.8
#omega_D = 2./3

# initial state(degrees)
theta1 = 0.
dtheta1 = 0.
theta2 = 60.
dtheta2 = 0.

n_steps = 3000
initial_n_phase_diagram = 0 #int(n_steps/2)
x0 = 0
xn = 500
x = np.linspace(x0, xn, n_steps+1)
dx = (xn - x0)/n_steps

y0 = np.radians([theta1, dtheta1, theta2, dtheta2])

# integrate your ODE using scipy.integrate.
y = integrate.odeint(derivs, y0, x)
sig = y[:,0].copy()

x1 = L1 * np.sin(y[:, 0])
y1 = -L1 * np.cos(y[:, 0])
x2 = x1 + L2 * np.sin(y[:, 2])
y2 = y1 - L2 * np.cos(y[:, 2])

phi = y.copy()
for j in range(n_steps):
    while phi[j,0]>np.pi or phi[j,0] < -np.pi:
        if(phi[j,0]>np.pi): phi[j,0]= phi[j,0]-2*np.pi
        if(phi[j,0]<-np.pi):phi[j,0]= phi[j,0]+2*np.pi


# fft
def baseline(y, tol=1.e-5):
    yy = y.copy()
    y_temp = np.zeros_like(yy)
    loop = 0
    while 1 :
        flag = 0
        for i in range(int(len(yy)/2)-1):
            yy[i+1] = np.minimum((yy[i]+yy[i+2])/2, yy[i+1])
            if np.abs(y_temp[i+1]-yy[i+1])>tol: flag=1
        y_temp = yy.copy()
        if flag <1: break
        loop += 1
        if loop > 100000: break
    return y_temp

n_samples = n_steps
sampling_time = xn
sampling_step = sampling_time/n_samples
cut_factor = 0.01
#time_vec = x

#FFT의 Power를 계산
# The FFT of the signal
sig_fft = fftpack.fft(sig)

# And the power (sig_fft is of complex dtype)
power = np.abs(sig_fft)
power_b = power.copy() 
power_base = baseline(power_b)

# The corresponding frequencies
#sample_freq = 2*np.pi*fftpack.fftfreq(sig.size, d=sampling_step)/omega_D
sample_freq = fftpack.fftfreq(sig.size, d=sampling_step)

power = power- (power_base)

# finding peak frequencies
pflag = 0
pos_mask = np.zeros_like(sample_freq, dtype=bool)
for i in range(1, len(power)-1):
    if(power[i]>power[i+1] and power[i]> cut_factor*power.max() 
       and sample_freq[i]>0. and pflag==1): 
        pos_mask[i] = True
        pflag = 0
        peak_freq, peak_power = sample_freq[pos_mask], power[pos_mask] 
    if(power[i]<power[i+1] and power[i]> cut_factor*power[:int(len(power)/4)].max() 
       and sample_freq[i]>0.): 
        pflag = 1
print(peak_freq)

# inverse fft 
inversed_sig = fftpack.ifft(sig_fft)




fig= plt.figure(figsize=(11,15))
plt.suptitle("Double Pendulum")

# double pendulum animation
ax = plt.subplot(321)
line, = ax.plot([],[], 'o-', lw=2)
plt.xlim(-3,3)
plt.ylim(-3,3)
ax.text(0.1, 0.9, "pendulum animation",transform=ax.transAxes)
ax.grid()
time_template = 'time = %.1f'
time_text = ax.text(0.75, 0.9, '', transform=ax.transAxes)

# phase diagram
ax2 = plt.subplot(322)
line2, = ax2.plot([],[], 'r,',lw=0.1)
plt.xlim(-np.pi, np.pi)
plt.ylim(-5, 5)
ax2.set_xlabel(r"$\phi$",verticalalignment='top')
ax2.set_ylabel(r"$\dot{\phi}$", rotation='horizontal')
ax2.text(0.1,0.9, 'phase diagram',transform=ax2.transAxes)
ax2.grid()
#freq_template = r' frequency = %.5f, period = %.5f in $T_D$'
#freq_text  =ax.text(0.55, 0.95, '', transform=ax.transAxes)

# time series
ax3 = plt.subplot(312)#ax = plt.gca()
plt.xlim(0, xn)
plt.ylim(1.5*y[:,0].min(), 1.5*y[:,0].max())
#param_text = ax.text(0.1,0.95, r"c=%.3f, F=%.4f, $\omega_D$=%.4f" 
#                     % (c,F,omega_D), transform=ax.transAxes)
param_text = ax3.text(0.05, 0.9, r"$\theta1$=%.1f, $\theta2$=%.1f" %(theta1, theta2), transform=ax3.transAxes)
line3,= plt.plot([], [],'r-', lw=0.5)
#line2, = plt.plot([], [], 'b-', lw=0.5)
ax3.grid()
ax3.set_ylabel(r"$\phi$",rotation='horizontal')
ax3.set_xlabel(r"$\tau$")
#time_template = 'time = %.1f'
#time_text = ax3.text(0.9, 0.9, '', transform=ax3.transAxes)


# Plot the FFT power - lower graph
ymax = 1.5*np.max(power)
plt.subplot(313)
ax4 = plt.gca()
xticks = np.arange(0, 5.25, 0.25)
ax4.set_xticks(xticks)
plt.plot(sample_freq[:int(n_samples/4)], power[:int(n_samples/4)], label='power spectrum')
#plt.text(0.1, 0.9, r'c=%.3f,  F=%.5f,  $\omega_D$=%.5f' 
#         % (c, F, omega_D), transform=ax4.transAxes)
plt.xlim(0,3)
# show peak frequencies on the peakes.
for j in range(len(peak_freq)):
    if(peak_power[j]>peak_power.max()/10.):
        plt.text(peak_freq[j], peak_power[j] + 10, '%.3f' % peak_freq[j])
plt.ylim(0,ymax)
#plt.xlabel(r'Frequency [$\omega_D/2\pi$]')
plt.xlabel(r'Frequency [Hz]')
plt.ylabel('power')
plt.grid()
plt.legend(loc='best')

def init():
    line.set_data([], [])
#    line2.set_data([], [])
    line2.set_data([], [])
    line3.set_data([],[])
    time_text.set_text('')
#    freq_text.set_text('')
#    return line, line2, line3, line4, time_text, freq_text
    return line, line2, line3, time_text

def animate(i):
    line.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
    if i>initial_n_phase_diagram:
        line2.set_data(phi[initial_n_phase_diagram:i,0], phi[initial_n_phase_diagram:i,1])
    line3.set_data(x[0:i], y[0:i,0])
#    line2.set_data(x[0:i], y_rk4[0:i,0])
    time_text.set_text(time_template % (i*dx))
#    freq_text.set_text(freq_template % (freq[i], period[i]))
#    return line, line2, line3, line4, time_text, freq_text
    return line, line2, line3, time_text

#file_name = (theta1 + theta2)
#writer_pillow = animation.PillowWriter(fps=20)

ani = animation.FuncAnimation(fig, animate, range(1, len(y)),
                              interval=dx, blit=True, init_func=init, repeat=False)
plt.show()
#ani.save('double pendulum.gif', writer=writer_pillow)
#plt.savefig(file_name + '_save.png')
