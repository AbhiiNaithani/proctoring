# import matplotlib.pyplot as plt
# import time
# import random
 
# xdata = []
# ydata = []
 
# plt.show()
 
# axes = plt.gca()
# axes.set_xlim(0, 100)
# axes.set_ylim(0,1)
# line, = axes.plot(xdata, ydata, 'r-')
 
# for i in range(100):
#     xdata.append(i)
#     ydata.append(i/2)
#     line.set_xdata(xdata)
#     line.set_ydata(ydata)
#     plt.draw()
#     plt.pause(1e-17)
 
# # add this if you don't want the window to disappear at the end
# plt.show()

# graph.py
# graph.py
import matplotlib
matplotlib.use('TkAgg')  # use this for Tkinter compatibility

import matplotlib.pyplot as plt
import time
from detection import get_score

def run_plot():
    plt.ion()
    fig, ax = plt.subplots()
    xdata, ydata = [], []
    line, = ax.plot([], [], 'r-', label='Cheat %')

    ax.set_title("Live Cheat Probability")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cheat Percentage")
    ax.set_xlim(0, 60)  # show last 60 seconds
    ax.set_ylim(0, 100)
    ax.legend()

    start_time = time.time()

    while True:
        current_time = time.time() - start_time
        current_score = get_score()

        xdata.append(current_time)
        ydata.append(current_score)

        # Keep last 60 seconds
        xdata = xdata[-300:]  # if updating every 0.2s, ~300 points = 60s
        ydata = ydata[-300:]

        line.set_data(xdata, ydata)
        ax.set_xlim(max(0, current_time - 60), current_time)
        ax.relim()
        ax.autoscale_view(scaley=True)

        plt.draw()
        plt.pause(0.2)

    plt.ioff()
    plt.show()

if __name__ == '__main__':
    run_plot()
