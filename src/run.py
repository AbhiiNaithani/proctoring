# import audio
# import head_pose
# import detection
# import threading as th


# if __name__ == "__main__":
#     # main()
#     head_pose_thread = th.Thread(target=head_pose.pose)
#     audio_thread = th.Thread(target=audio.sound)
#     detection_thread = th.Thread(target=detection.run_detection)

#     head_pose_thread.start()
#     audio_thread.start()
#     detection_thread.start()

#     head_pose_thread.join()
#     audio_thread.join()
#     detection_thread.join()

import cv2
import threading
import audio
import head_pose
import detection
import phone_detection
import combined_view
import time
from queue import Queue
import graph

def main():
    # Camera setup
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    
    if not cap.isOpened():
        print("Camera not accessible")
        return

    # Shared resources
    frame_queue = Queue(maxsize=1)
    stop_event = threading.Event()

    # Frame capture thread
    def frame_capture():
        while not stop_event.is_set():
            ret, frame = cap.read()
            if ret:
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()  # Discard oldest frame if queue is full
                    except:
                        pass
                frame_queue.put(frame.copy())
            time.sleep(0.01)  # Small sleep to prevent CPU overload

    # Start GUI-related function on main thread FIRST
    detection_thread = threading.Thread(target=detection.run_detection,args=(stop_event,))
    detection_thread.start()

    # Start all components
    threads = [
        threading.Thread(target=frame_capture, daemon=True),
        threading.Thread(target=audio.sound, daemon=True),
        # threading.Thread(target=head_pose.pose, args=(frame_queue, stop_event), daemon=True),
        # threading.Thread(target=phone_detection.detect_phone, args=(frame_queue, stop_event), daemon=True),
        threading.Thread(target=detection.run_detection, daemon=True),
        # threading.Thread(target=combined_view.process_combined_view, args=(frame_queue, stop_event), daemon=True),
        threading.Thread(target=graph.run_plot, args=(stop_event,), daemon=True)
    ]

    for t in threads:
        t.start()

    # Clean exit handling
    try:
        # while detection_thread.is_alive():
        #     time.sleep(0.5)
        combined_view.process_combined_view(frame_queue, stop_event)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stop_event.set()
        time.sleep(1)
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()