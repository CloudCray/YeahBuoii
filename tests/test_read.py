__author__ = 'Cloud'
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pylab import *
from threading import Thread

from models.buoy import Buoy

COM_PORT = "COM3"
BAUD = 9600
max_min_val = 6000

buoy = Buoy(COM_PORT, BAUD)

def buoy_reader():
    while True:
        buoy.read_next()

thd = Thread(target=buoy_reader)
thd.start()

ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-max_min_val, max_min_val])
ax.set_ylim([-max_min_val, max_min_val])
ax.set_zlim([-max_min_val, max_min_val])


def get_acc_data(b):
    x = b.values[1]
    y = b.values[2]
    z = b.values[3]
    return x, y, z

# Photo1
photo_1_fig = plt.figure()
photo_2_fig = plt.figure()
p1_ax = photo_1_fig.add_subplot(111)
p2_ax = photo_2_fig.add_subplot(111)
p1_ax.set_ylim([0, 1024])
p2_ax.set_ylim([0, 1024])
p1_res = None
p2_res = None
p1_last_100 = []
p2_last_100 = []

temp_fig = plt.figure()
temp_ax = temp_fig.add_subplot(111)
temp_ax.set_ylim([0, 35])
temp_res = None
temp_last_100 = []

res = None
plotting = True

last_10 = []

while plotting:
    data = get_acc_data(buoy)
    last_10.append(data)
    if len(last_10) > 10:
        last_10.pop(0)
    xs = np.array([x[0] for x in last_10])
    ys = np.array([x[1] for x in last_10])
    zs = np.array([x[2] for x in last_10])
    if res:
        res.remove()
    res = ax.scatter(xs, ys, zs)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')



    # P1
    x = buoy.values[4]
    p1_last_100.append(x)
    if len(p1_last_100) > 100:
        p1_last_100.pop(0)
    xs = np.array(p1_last_100)

    # P2
    x2 = buoy.values[5]
    p2_last_100.append(x)
    if len(p2_last_100) > 100:
        p2_last_100.pop(0)
    xs2 = np.array(p2_last_100)

    if p1_res:
        p1_res.pop(0).remove()
    if p2_res:
        p2_res.pop(0).remove()
    p1_res = p1_ax.plot(xs, color='b', fillstyle='full', linewidth='10')
    p2_res = p2_ax.plot(xs2, color='g', fillstyle='full', linewidth='10')

    # Temp
    t = buoy.values[0]
    temp_last_100.append(t)
    if len(temp_last_100) > 100:
        temp_last_100.pop(0)
    ts = np.array(temp_last_100)
    if temp_res:
        temp_res.pop(0).remove()
    temp_res = temp_ax.plot(ts, color='r', fillstyle='full', linewidth='10')

    draw()
    pause(0.01)
