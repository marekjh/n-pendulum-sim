import compute as com
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import pickle
import sys
import os

# Parse command line args
curve = False
filename = "data.pickle"
for e in sys.argv[1:]:
    if e.endswith(".pickle"):
        filename = e
    elif e == "--curve":
        curve = True
with open(os.path.join("saved", filename), "rb") as f:
    theta1, theta2, params = pickle.load(f)

# Convert angle data into (x, y) data
x1 = com.L1 * np.sin(theta1)
x2 = x1 + com.L2 * np.sin(theta2)
y1 = -com.L1 * np.cos(theta1)
y2 = y1 - com.L2 * np.cos(theta2)

# Compute/print system properties
bound = com.L1 + com.L2
(m1, m2, l1, l2, g, th1_0, th2_0, th1d_0, th2d_0) = params
print(f"m1 = {m1} kg\nm2 = {m2} kg\nl1 = {l1} m\nl2 = {l2} m\ng = {g} m/s^2\ntheta1_initial = {th1_0} deg\n" +
      f"theta2_initial = {th2_0} deg\nomega1_initial = {th1d_0} deg/s\nomega2_initial = {th2d_0} deg/s")

# Draw pendulum config to screen once per time step
fig = plt.figure()
def draw_plot(i):
    plt.clf()
    x1i, x2i, y1i, y2i = x1[i], x2[i], y1[i], y2[i]
    plt.scatter(x1i, y1i, c="red")
    plt.scatter(x2i, y2i, c="red")
    plt.plot([0, x1i], [0, y1i], c="black")
    plt.plot([x1i, x2i], [y1i, y2i], c="black")
    plt.xlim(-bound, bound)
    plt.ylim(-bound, bound)
    plt.title(f"t={round(com.t[i], 3)}")

if curve:
    plt.plot(x2, y2)
else:
    animator = ani.FuncAnimation(fig, draw_plot, range(len(theta1)), interval=com.interval)
plt.show()
