import matplotlib

matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

r, T, t = 0.5, 5, 0.01
a, b = 7.25, 1.25

w = 2 * np.pi / T
v = w * r
x, y, th = 0., 0., 0.
traj_x, traj_y = [], []

v2, w2 = 0., 0.
x2, y2, th2 = -0.1, -0.1, 0.
traj_x2, traj_y2 = [], []

fig = plt.figure(0)
ax = fig.add_subplot()
pos, = ax.plot(x, y, 'ro')
pos2, = ax.plot(x2, y2, 'bo')
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def update(i):
    global x, y, th, v, w
    global x2, y2, th2, v2, w2

    # eight-shape
    th = (th + w * t) % (2 * np.pi)
    x += v * t * np.cos(th)
    y += v * t * np.sin(th)
    
    if abs(th - 2 * np.pi) <= 0.05:
        w *= -1
    
    # follower
    dist = np.sqrt((x - x2) ** 2 + (y - y2) ** 2)
    v2 = a * dist
    w2 = b * np.sign(th - th2) * abs(np.arctan2(y - y2, x - x2))
    if dist <= 0.05:
        v2 = 0.
    if abs(th - th2) <= 0.05:
        w2 = 0.
    th2 = (th2 + w2 * t) % (2 * np.pi)
    x2 += v2 * t * np.cos(th2)
    y2 += v2 * t * np.sin(th2)
    
    # visualization
    traj_x.append(x)
    traj_y.append(y)
    traj_x2.append(x2)
    traj_y2.append(y2)
    
    pos.set_data(x, y)
    pos2.set_data(x2, y2)
    ax.plot(traj_x, traj_y, 'r-', lw=0.1)
    ax.plot(traj_x2, traj_y2, 'b-', lw=0.1)
    
    now_time = time.time()
    interval = now_time - base_time
    time_text.set_text('t = %.2fs th = %.3f th2 = %.3f dist = %.3fm' % (i * t, th, th2, dist))
    with open("data.txt", 'a') as f:
        f.write("%.3f %.3f %.3f\n" % (i * t, th - th2, dist))
    
    return pos, pos2, time_text


ani = animation.FuncAnimation(fig, update, interval=10)
base_time = time.time()
plt.xlabel("X/m")
plt.ylabel("Y/m")
plt.show()
