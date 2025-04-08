# import cv2
# import torch
# import numpy as np
# import warnings

# # Suppress all warnings
# warnings.filterwarnings("ignore")

# # Global cheat flag for phone detection
# PHONE_CHEAT = 0

# # Load YOLOv5s model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, autoshape=True, verbose=False)
# model.conf = 0.4  # confidence threshold
# model.classes = [67]  # Only detect cell phones (class 67 in COCO)
# model.eval()  # Set to evaluation mode

# def detect_phone(frame_queue, stop_event):
#     global PHONE_CHEAT
    
#     # Warm up model
#     dummy_input = torch.zeros((1, 3, 640, 640))
#     with torch.no_grad():
#         _ = model(dummy_input)
    
#     while not stop_event.is_set():
#         if not frame_queue.empty():
#             frame = frame_queue.get()
            
#             # Run inference
#             with torch.no_grad(), torch.inference_mode():
#                 results = model(frame)
                
#                 # Process detections
#                 detections = results.pandas().xyxy[0]
#                 PHONE_CHEAT = 1 if len(detections) > 0 else 0
                
#                 # Draw bounding boxes if phone detected
#                 if PHONE_CHEAT:
#                     for _, det in detections.iterrows():
#                         x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
#                         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                         cv2.putText(frame, f"Phone {det['confidence']:.2f}", 
#                                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
#                                    0.5, (0, 0, 255), 2)
#                     print("Phone Detected!")
                
#                 # Display frame (optional)
#                 cv2.imshow('Phone Detection', frame)
#                 if cv2.waitKey(1) & 0xFF == 27:
#                     stop_event.set()
#                     break

#     cv2.destroyAllWindows()



import cv2
import torch
import numpy as np
import warnings
import time

# Suppress all warnings
warnings.filterwarnings("ignore")

# Global cheat flag for phone detection
PHONE_CHEAT = 0

# Load medium-sized YOLOv5 model (small version)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, autoshape=True, verbose=False)
model.conf = 0.4  # Confidence threshold (lower than before to improve detection)
model.classes = [67]  # Only detect cell phones (class 67 in COCO)
model.eval()  # Set to evaluation mode

def detect_phone(frame_queue, stop_event):
    global PHONE_CHEAT
    
    # Warm up model with proper input size
    dummy_input = torch.zeros((1, 3, 640, 640))
    with torch.no_grad():
        _ = model(dummy_input)
    
    # Variables for FPS calculation and frame skipping
    frame_count = 0
    start_time = time.time()
    
    while not stop_event.is_set():
        if not frame_queue.empty():
            try:
                frame = frame_queue.get()
                
                # Only process every 3rd frame to improve performance
                frame_count += 1
                if frame_count % 3 != 0:
                    continue
                
                # Run inference on the full frame (no resizing)
                with torch.no_grad():
                    results = model(frame)
                
                # Process detections
                detections = results.pandas().xyxy[0]
                PHONE_CHEAT = 1 if len(detections) > 0 else 0
                
                # Draw bounding boxes if phone detected
                if PHONE_CHEAT:
                    for _, det in detections.iterrows():
                        x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f"Phone {det['confidence']:.2f}", 
                                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.5, (0, 0, 255), 2)
                    print('Phone Detected!!')
                
                # Calculate and display FPS
                fps = frame_count / (time.time() - start_time)
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow('Phone Detection', frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    stop_event.set()
                    break
                    
            except Exception as e:
                print(f"Error processing frame: {e}")
                continue

    cv2.destroyAllWindows()