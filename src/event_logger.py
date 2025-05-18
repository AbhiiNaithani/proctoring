
# import os
# import csv
# import cv2
# import time
# from datetime import datetime

# LOG_DIR = "session_logs"
# IMG_DIR = os.path.join(LOG_DIR, "images")
# CSV_PATH = os.path.join(LOG_DIR, "events.csv")
# MAX_ENTRIES = 1000
# LOG_INTERVAL = 0.5 # seconds


# # Ensure folders
# os.makedirs(IMG_DIR, exist_ok=True)

# # Create CSV with headers (if missing)
# if not os.path.exists(CSV_PATH):
#     with open(CSV_PATH, mode='w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow([
#             "timestamp", "looking_left", "looking_right", "looking_up", "looking_down",
#             "phone_detected", "multiple_faces", "image_path"
#         ])

# # Internal logger state
# last_log_time = 0
# logged_entries = 0

# def reset_logger_state():
#     global last_log_time, logged_entries
#     last_log_time = 0
#     logged_entries = 0

# def log_event(frame, looking_left, looking_right, looking_up, looking_down, phone_detected, multiple_faces):
#     global last_log_time, logged_entries

#     now = time.time()
#     if logged_entries >= MAX_ENTRIES:
#         return  # Stop logging after N entries

#     if now - last_log_time < LOG_INTERVAL:
#         return  # Throttle logging

#     last_log_time = now

#     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
#     img_filename = f"{timestamp}.jpg"
#     img_path = os.path.join(IMG_DIR, img_filename)

#     # Save frame
#     cv2.imwrite(img_path, frame)

#     # Log to CSV
#     with open(CSV_PATH, mode='a', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow([
#             timestamp,
#             int(looking_left),
#             int(looking_right),
#             int(looking_up),
#             int(looking_down),
#             int(phone_detected),
#             int(multiple_faces),
#             img_path
#         ])

#     logged_entries += 1


import os
import csv
import cv2
import time
from datetime import datetime

LOG_DIR = "session_logs"
IMG_DIR = os.path.join(LOG_DIR, "images")
MAX_ENTRIES = 500
LOG_INTERVAL = 0.5  # seconds

# Internal logger state
last_log_time = 0
logged_entries = 0
current_session_csv = None

def get_session_filename():
    """Generate a unique filename for each session"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(LOG_DIR, f"session_{timestamp}.csv")

def init_new_session():
    """Initialize a new logging session"""
    global current_session_csv, last_log_time, logged_entries
    
    # Ensure folders exist
    os.makedirs(IMG_DIR, exist_ok=True)
    
    # Create new CSV file
    current_session_csv = get_session_filename()
    last_log_time = 0
    logged_entries = 0
    
    # Write CSV headers
    with open(current_session_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "looking_left", "looking_right", "looking_up", "looking_down",
            "phone_detected", "multiple_faces", "image_path"
        ])

def log_event(frame, looking_left, looking_right, looking_up, looking_down, 
              phone_detected, multiple_faces):
    global last_log_time, logged_entries

    # Initialize new session if needed
    if current_session_csv is None:
        init_new_session()

    now = time.time()
    if logged_entries >= MAX_ENTRIES:
        return  # Stop logging after N entries

    if now - last_log_time < LOG_INTERVAL:
        return  # Throttle logging

    last_log_time = now

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    img_filename = f"{timestamp}.jpg"
    img_path = os.path.join(IMG_DIR, img_filename)

    # Save frame
    cv2.imwrite(img_path, frame)

    # Log to CSV
    with open(current_session_csv, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            int(looking_left),
            int(looking_right),
            int(looking_up),
            int(looking_down),
            int(phone_detected),
            int(multiple_faces),
            img_path
        ])

    logged_entries += 1