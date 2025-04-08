# import time
# import audio
# import head_pose
# import matplotlib.pyplot as plt
# import numpy as np

# PLOT_LENGTH = 200

# # place holders 
# GLOBAL_CHEAT = 0
# PERCENTAGE_CHEAT = 0
# CHEAT_THRESH = 0.6
# XDATA = list(range(200))
# YDATA = [0]*200

# def avg(current, previous):
#     if previous > 1:
#         return 0.65
#     if current == 0:
#         if previous < 0.01:
#             return 0.01
#         return previous / 1.01
#     if previous == 0:
#         return current
#     return 1 * previous + 0.1 * current

# def process():
#     global GLOBAL_CHEAT, PERCENTAGE_CHEAT, CHEAT_THRESH #head_pose.X_AXIS_CHEAT, head_pose.Y_AXIS_CHEAT, audio.AUDIO_CHEAT
#     # print(head_pose.X_AXIS_CHEAT, head_pose.Y_AXIS_CHEAT)
#     # print("entered proess()...")
#     if GLOBAL_CHEAT == 0:
#         if head_pose.X_AXIS_CHEAT == 0:
#             if head_pose.Y_AXIS_CHEAT == 0:
#                 if audio.AUDIO_CHEAT == 0:
#                     PERCENTAGE_CHEAT = avg(0, PERCENTAGE_CHEAT)
#                 else:
#                     PERCENTAGE_CHEAT = avg(0.2, PERCENTAGE_CHEAT)
#             else:
#                 if audio.AUDIO_CHEAT == 0:
#                     PERCENTAGE_CHEAT = avg(0.2, PERCENTAGE_CHEAT)
#                 else:
#                     PERCENTAGE_CHEAT = avg(0.4, PERCENTAGE_CHEAT)
#         else:
#             if head_pose.Y_AXIS_CHEAT == 0:
#                 if audio.AUDIO_CHEAT == 0:
#                     PERCENTAGE_CHEAT = avg(0.1, PERCENTAGE_CHEAT)
#                 else:
#                     PERCENTAGE_CHEAT = avg(0.4, PERCENTAGE_CHEAT)
#             else:
#                 if audio.AUDIO_CHEAT == 0:
#                     PERCENTAGE_CHEAT = avg(0.15, PERCENTAGE_CHEAT)
#                 else:
#                     PERCENTAGE_CHEAT = avg(0.25, PERCENTAGE_CHEAT)
#     else:
#         if head_pose.X_AXIS_CHEAT == 0:
#             if head_pose.Y_AXIS_CHEAT == 0:
#                 if audio.AUDIO_CHEAT == 0:
#                     PERCENTAGE_CHEAT = avg(0, PERCENTAGE_CHEAT)
#                 else:
#                     PERCENTAGE_CHEAT = avg(0.55, PERCENTAGE_CHEAT)
#             else:
#                 if audio.AUDIO_CHEAT == 0:
#                     PERCENTAGE_CHEAT = avg(0.55, PERCENTAGE_CHEAT)
#                 else:
#                     PERCENTAGE_CHEAT = avg(0.85, PERCENTAGE_CHEAT)
#         else:
#             if head_pose.Y_AXIS_CHEAT == 0:
#                 if audio.AUDIO_CHEAT == 0:
#                     PERCENTAGE_CHEAT = avg(0.6, PERCENTAGE_CHEAT)
#                 else:
#                     PERCENTAGE_CHEAT = avg(0.85, PERCENTAGE_CHEAT)
#             else:
#                 if audio.AUDIO_CHEAT == 0:
#                     PERCENTAGE_CHEAT = avg(0.5, PERCENTAGE_CHEAT)
#                 else:
#                     PERCENTAGE_CHEAT = avg(0.85, PERCENTAGE_CHEAT)

#     if PERCENTAGE_CHEAT > CHEAT_THRESH:
#         GLOBAL_CHEAT = 1
#         print("CHEATING")
#     else:
#         GLOBAL_CHEAT = 0
#     print("Cheat percent: ", PERCENTAGE_CHEAT, GLOBAL_CHEAT)

# def run_detection():
#     global XDATA,YDATA
#     plt.show()
#     axes = plt.gca()
#     axes.set_xlim(0, 200)
#     axes.set_ylim(0,1)
#     line, = axes.plot(XDATA, YDATA, 'r-')
#     plt.title("SUSpicious Behaviour Detection")
#     plt.xlabel("Time")
#     plt.ylabel("Cheat Probablity")
#     i = 0
#     while True:
#         YDATA.pop(0)
#         YDATA.append(PERCENTAGE_CHEAT)
#         line.set_xdata(XDATA)
#         line.set_ydata(YDATA)
#         plt.draw()
#         plt.pause(1e-17)
#         time.sleep(1/5)
#         process()

import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better compatibility

import time
import audio
import head_pose
import phone_detection
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading


# Constants
PLOT_LENGTH = 200
CHEAT_THRESH = 0.6

# State variables
GLOBAL_CHEAT = 0
PERCENTAGE_CHEAT = 0
XDATA = list(range(PLOT_LENGTH))
YDATA = [0] * PLOT_LENGTH

# Initialize plot
# plt.ion()
# fig, ax = plt.subplots(figsize=(10, 5))
# line, = ax.plot(XDATA, YDATA, 'r-')
# ax.set_xlim(0, PLOT_LENGTH)
# ax.set_ylim(0, 1)
# ax.set_title("Real-time Cheat Detection", fontsize=14)
# ax.set_xlabel("Time (frames)", fontsize=12)
# ax.set_ylabel("Cheat Probability", fontsize=12)
# ax.grid(True)
# fig.tight_layout()

# def update_plot(frame):
#     line.set_ydata(YDATA)
#     return line,

cheat_score = 0
lock = threading.Lock()

def set_score(value):
    with lock:
        global cheat_score
        cheat_score = value

def get_score():
    with lock:
        return cheat_score
    
def avg(current, previous):
    if previous > 1:
        return 0.65
    if current == 0:
        if previous < 0.01:
            return 0.01
        return previous / 1.01
    if previous == 0:
        return current
    return 1 * previous + 0.1 * current


# def calculate_cheat():
#     """Calculate cheat probability based on all detectors"""
#     base = 0.0
#     if head_pose.X_AXIS_CHEAT: base += 0.15
#     if head_pose.Y_AXIS_CHEAT: base += 0.15
#     if audio.AUDIO_CHEAT: base += 0.25
#     if phone_detection.PHONE_CHEAT: base += 0.35
#     return min(base, 1.0)  # Cap at 100%

# def process():
#     global GLOBAL_CHEAT, PERCENTAGE_CHEAT
    
#     # current_cheat = calculate_cheat()
#     # PERCENTAGE_CHEAT = weighted_avg(current_cheat, PERCENTAGE_CHEAT)
#     phone_flag = phone_detection.PHONE_CHEAT

#     if GLOBAL_CHEAT == 0:
#         if head_pose.X_AXIS_CHEAT == 0:
#             if head_pose.Y_AXIS_CHEAT == 0:
#                 if audio.AUDIO_CHEAT == 0:
#                     if phone_flag == 0:
#                         PERCENTAGE_CHEAT = avg(0, PERCENTAGE_CHEAT)
#                     else:
#                         PERCENTAGE_CHEAT = avg(0.4, PERCENTAGE_CHEAT)
#                 else:
#                     PERCENTAGE_CHEAT = avg(0.2 + 0.2 * phone_flag, PERCENTAGE_CHEAT)
#             else:
#                 if audio.AUDIO_CHEAT == 0:
#                     PERCENTAGE_CHEAT = avg(0.2 + 0.2 * phone_flag, PERCENTAGE_CHEAT)
#                 else:
#                     PERCENTAGE_CHEAT = avg(0.4 + 0.2 * phone_flag, PERCENTAGE_CHEAT)
#         else:
#             if head_pose.Y_AXIS_CHEAT == 0:
#                 if audio.AUDIO_CHEAT == 0:
#                     PERCENTAGE_CHEAT = avg(0.1 + 0.2 * phone_flag, PERCENTAGE_CHEAT)
#                 else:
#                     PERCENTAGE_CHEAT = avg(0.4 + 0.2 * phone_flag, PERCENTAGE_CHEAT)
#             else:
#                 if audio.AUDIO_CHEAT == 0:
#                     PERCENTAGE_CHEAT = avg(0.15 + 0.2 * phone_flag, PERCENTAGE_CHEAT)
#                 else:
#                     PERCENTAGE_CHEAT = avg(0.25 + 0.2 * phone_flag, PERCENTAGE_CHEAT)
#     else:
#         if head_pose.X_AXIS_CHEAT == 0:
#             if head_pose.Y_AXIS_CHEAT == 0:
#                 if audio.AUDIO_CHEAT == 0:
#                     if phone_flag == 0:
#                         PERCENTAGE_CHEAT = avg(0, PERCENTAGE_CHEAT)
#                     else:
#                         PERCENTAGE_CHEAT = avg(0.55, PERCENTAGE_CHEAT)
#                 else:
#                     PERCENTAGE_CHEAT = avg(0.55 + 0.2 * phone_flag, PERCENTAGE_CHEAT)
#             else:
#                 if audio.AUDIO_CHEAT == 0:
#                     PERCENTAGE_CHEAT = avg(0.55 + 0.2 * phone_flag, PERCENTAGE_CHEAT)
#                 else:
#                     PERCENTAGE_CHEAT = avg(0.85 + 0.2 * phone_flag, PERCENTAGE_CHEAT)
#         else:
#             if head_pose.Y_AXIS_CHEAT == 0:
#                 if audio.AUDIO_CHEAT == 0:
#                     PERCENTAGE_CHEAT = avg(0.6 + 0.2 * phone_flag, PERCENTAGE_CHEAT)
#                 else:
#                     PERCENTAGE_CHEAT = avg(0.85 + 0.2 * phone_flag, PERCENTAGE_CHEAT)
#             else:
#                 if audio.AUDIO_CHEAT == 0:
#                     PERCENTAGE_CHEAT = avg(0.5 + 0.2 * phone_flag, PERCENTAGE_CHEAT)
#                 else:
#                     PERCENTAGE_CHEAT = avg(0.85 + 0.2 * phone_flag, PERCENTAGE_CHEAT)


#     # Update data for plot
#     YDATA.pop(0)
#     YDATA.append(PERCENTAGE_CHEAT)
    
#     # Check threshold
#     GLOBAL_CHEAT = 1 if PERCENTAGE_CHEAT > CHEAT_THRESH else 0
#     status = "CHEATING!" if GLOBAL_CHEAT else "Normal"
#     print(f"Cheat: {PERCENTAGE_CHEAT:.4f} | Status: {status}")

# def run_detection():
#     # Start animation
#     ani = FuncAnimation(fig, update_plot, interval=200, blit=True, cache_frame_data=False)
    
#     # Main processing loop
#     try:
#         while True:
#             process()
#             plt.pause(0.1)  # Allow time for graph updates
#     except KeyboardInterrupt:
#         plt.close('all')
def process():
    global GLOBAL_CHEAT, PERCENTAGE_CHEAT
    
    phone_flag = phone_detection.PHONE_CHEAT
    phone_weight = 0.5 if phone_flag else 0  # Higher base weight for phone detection
    
    if GLOBAL_CHEAT == 0:
        if head_pose.X_AXIS_CHEAT == 0:
            if head_pose.Y_AXIS_CHEAT == 0:
                if audio.AUDIO_CHEAT == 0:
                    PERCENTAGE_CHEAT = avg(phone_weight, PERCENTAGE_CHEAT)
                else:
                    PERCENTAGE_CHEAT = avg(0.3 + phone_weight, PERCENTAGE_CHEAT)
            else:
                if audio.AUDIO_CHEAT == 0:
                    PERCENTAGE_CHEAT = avg(0.3 + phone_weight, PERCENTAGE_CHEAT)
                else:
                    PERCENTAGE_CHEAT = avg(0.5 + phone_weight, PERCENTAGE_CHEAT)
        else:
            if head_pose.Y_AXIS_CHEAT == 0:
                if audio.AUDIO_CHEAT == 0:
                    PERCENTAGE_CHEAT = avg(0.2 + phone_weight, PERCENTAGE_CHEAT)
                else:
                    PERCENTAGE_CHEAT = avg(0.5 + phone_weight, PERCENTAGE_CHEAT)
            else:
                if audio.AUDIO_CHEAT == 0:
                    PERCENTAGE_CHEAT = avg(0.25 + phone_weight, PERCENTAGE_CHEAT)
                else:
                    PERCENTAGE_CHEAT = avg(0.35 + phone_weight, PERCENTAGE_CHEAT)
    else:
        # When already in cheating state, phone detection has even greater impact
        phone_weight = 0.7 if phone_flag else 0
        
        if head_pose.X_AXIS_CHEAT == 0:
            if head_pose.Y_AXIS_CHEAT == 0:
                if audio.AUDIO_CHEAT == 0:
                    PERCENTAGE_CHEAT = avg(phone_weight, PERCENTAGE_CHEAT)
                else:
                    PERCENTAGE_CHEAT = avg(0.6 + phone_weight*0.3, PERCENTAGE_CHEAT)
            else:
                if audio.AUDIO_CHEAT == 0:
                    PERCENTAGE_CHEAT = avg(0.6 + phone_weight*0.3, PERCENTAGE_CHEAT)
                else:
                    PERCENTAGE_CHEAT = avg(0.9, PERCENTAGE_CHEAT)  # Max out when phone + other factors
        else:
            if head_pose.Y_AXIS_CHEAT == 0:
                if audio.AUDIO_CHEAT == 0:
                    PERCENTAGE_CHEAT = avg(0.7 + phone_weight*0.2, PERCENTAGE_CHEAT)
                else:
                    PERCENTAGE_CHEAT = avg(0.9, PERCENTAGE_CHEAT)
            else:
                if audio.AUDIO_CHEAT == 0:
                    PERCENTAGE_CHEAT = avg(0.6 + phone_weight*0.3, PERCENTAGE_CHEAT)
                else:
                    PERCENTAGE_CHEAT = avg(0.95, PERCENTAGE_CHEAT)  # Near certain cheating

    # Update data for plot
    YDATA.pop(0)
    YDATA.append(PERCENTAGE_CHEAT)

    set_score(PERCENTAGE_CHEAT)
    
    # Check threshold (lowered slightly to account for increased weights)
    GLOBAL_CHEAT = 1 if PERCENTAGE_CHEAT > CHEAT_THRESH else 0
    status = "CHEATING!" if GLOBAL_CHEAT else "Normal"
    print(f"Cheat: {PERCENTAGE_CHEAT:.4f} | Status: {status} | Phone: {'Yes' if phone_flag else 'No'}")
def run_detection():
    # Initialize plot with clearer settings
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    line, = ax.plot(XDATA, YDATA, 'r-')
    ax.set_xlim(0, PLOT_LENGTH)
    ax.set_ylim(0, 1)
    ax.set_title("Real-time Cheat Detection", fontsize=14)
    ax.set_xlabel("Time (frames)", fontsize=12)
    ax.set_ylabel("Cheat Probability", fontsize=12)
    ax.grid(True)
    fig.tight_layout()
    fig.canvas.draw()
    
    try:
        while True:
            process()
            # Update plot
            line.set_ydata(YDATA)
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.05)  # 20 updates per second
            
    except KeyboardInterrupt:
        plt.close('all')
