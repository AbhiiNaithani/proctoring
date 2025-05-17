from audioop import avg
from glob import glob
from itertools import count
import cv2
import mediapipe as mp
import numpy as np
import threading as th
import sounddevice as sd
import audio

# Global variables
x = 0  # X axis head pose (for primary face)
y = 0  # Y axis head pose (for primary face)

X_AXIS_CHEAT = 0
Y_AXIS_CHEAT = 0
MULTIPLE_FACE_CHEAT = 0

def pose(frame_queue, stop_event):
    global VOLUME_NORM, x, y, X_AXIS_CHEAT, Y_AXIS_CHEAT, MULTIPLE_FACE_CHEAT
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_faces=5  # Increase maximum number of detectable faces
    )
    
    mp_drawing = mp.solutions.drawing_utils

    while not stop_event.is_set():
        if not frame_queue.empty():
            image = frame_queue.get()
            if image is None:
                print("❌ Failed to grab frame from camera.")
                continue

            # Flip the image horizontally for a later selfie-view display
            # Also convert the color space from BGR to RGB
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            # To improve performance
            image.flags.writeable = False
            
            # Get the result
            results = face_mesh.process(image)
            
            # To improve performance
            image.flags.writeable = True
            
            # Convert the color space from RGB to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            img_h, img_w, img_c = image.shape
            
            # Track if we've processed the primary face
            primary_face_processed = False
            
            if results.multi_face_landmarks:
                # Update multiple face detection flag
                MULTIPLE_FACE_CHEAT = 1 if len(results.multi_face_landmarks) > 1 else 0
                
                for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
                    face_ids = [33, 263, 1, 61, 291, 199]
                    face_3d = []
                    face_2d = []
                    
                    # Draw landmarks for all faces
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None)
                    
                    # Label each face
                    cv2.putText(image, f"Face {face_idx+1}", 
                               (20, 40 + face_idx * 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, (0, 255, 0), 2)
                    
                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx in face_ids:
                            if idx == 1:
                                nose_2d = (lm.x * img_w, lm.y * img_h)
                                nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                            x, y = int(lm.x * img_w), int(lm.y * img_h)

                            # Get the 2D Coordinates
                            face_2d.append([x, y])

                            # Get the 3D Coordinates
                            face_3d.append([x, y, lm.z])       
                    
                    # Only calculate pose for the primary face (face_idx == 0)
                    if face_idx == 0 and len(face_2d) > 0 and len(face_3d) > 0:
                        # Convert to NumPy arrays
                        face_2d = np.array(face_2d, dtype=np.float64)
                        face_3d = np.array(face_3d, dtype=np.float64)

                        # The camera matrix
                        focal_length = 1 * img_w
                        cam_matrix = np.array([
                            [focal_length, 0, img_h / 2],
                            [0, focal_length, img_w / 2],
                            [0, 0, 1]
                        ])

                        # The Distance Matrix
                        dist_matrix = np.zeros((4, 1), dtype=np.float64)

                        # Solve PnP
                        success, rot_vec, trans_vec = cv2.solvePnP(
                            face_3d, face_2d, cam_matrix, dist_matrix)

                        # Get rotational matrix
                        rmat, jac = cv2.Rodrigues(rot_vec)

                        # Get angles
                        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                        # Get the rotation degrees
                        x = angles[0] * 360
                        y = angles[1] * 360

                        # Determine head direction
                        if y < -10:
                            text = "Looking Left"
                        elif y > 10:
                            text = "Looking Right"
                        elif x < -10:
                            text = "Looking Down"
                        else:
                            text = "Forward"
                            
                        text = f"Primary Face: {text} (X:{int(x)} Y:{int(y)})"
                        
                        # Update cheat flags for primary face only
                        if y < -10 or y > 10:
                            X_AXIS_CHEAT = 1
                        else:
                            X_AXIS_CHEAT = 0

                        if x < -5:
                            Y_AXIS_CHEAT = 1
                        else:
                            Y_AXIS_CHEAT = 0

                        # Display the primary face direction
                        cv2.putText(image, text, (20, 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.7, (0, 0, 255), 2)
                        
                        # Mark that we've processed the primary face
                        primary_face_processed = True
            
            # If no faces detected, reset flags
            else:
                MULTIPLE_FACE_CHEAT = 0
                X_AXIS_CHEAT = 0
                Y_AXIS_CHEAT = 0
            
            # Display the image
            cv2.imshow('Head Pose Estimation', image)

            if cv2.waitKey(5) & 0xFF == 27:
                stop_event.set()
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    t1 = th.Thread(target=pose)
    t1.start()
    t1.join()