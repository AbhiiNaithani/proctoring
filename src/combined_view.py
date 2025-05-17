import cv2
import torch
import numpy as np
import mediapipe as mp
import head_pose
import phone_detection

def process_combined_view(frame_queue, stop_event):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5,  # Increased from 0.3 for better accuracy
        min_tracking_confidence=0.5,
        max_num_faces=5  # Explicitly set max faces
    )
    mp_drawing = mp.solutions.drawing_utils

    while not stop_event.is_set():
        if not frame_queue.empty():
            frame = frame_queue.get()
            annotated_frame = frame.copy()

            # Head Pose Estimation
            img_h, img_w, _ = frame.shape
            image_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)
            face_ids = [33, 263, 1, 61, 291, 199]

            if results.multi_face_landmarks:
                face_count = len(results.multi_face_landmarks)
                head_pose.MULTIPLE_FACE_CHEAT = 1 if face_count > 1 else 0
                
                # Display face count
                cv2.putText(annotated_frame, f"Faces: {face_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                for i, landmarks in enumerate(results.multi_face_landmarks):
                    # Draw landmarks for all faces
                    mp_drawing.draw_landmarks(
                        annotated_frame, 
                        landmarks, 
                        mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(
                            color=(0, 255, 0) if i == 0 else (0, 0, 255),  # Primary face green, others red
                            thickness=1,
                            circle_radius=1
                        )
                    )
                    
                    # Label each face
                    cv2.putText(annotated_frame, f"Face {i+1}", (10, 60 + i*20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                (0, 255, 0) if i == 0 else (0, 0, 255), 2)

                    # Only process head pose for primary face (i == 0)
                    if i == 0:
                        face_3d = []
                        face_2d = []
                        
                        for idx, lm in enumerate(landmarks.landmark):
                            if idx in face_ids:
                                x, y = int(lm.x * img_w), int(lm.y * img_h)
                                face_2d.append([x, y])
                                face_3d.append([x, y, lm.z])
                                if idx == 1:
                                    nose_2d = (x, y)
                                    nose_3d = (x, y, lm.z * 8000)

                        if len(face_2d) > 0 and len(face_3d) > 0:
                            face_2d = np.array(face_2d, dtype=np.float64)
                            face_3d = np.array(face_3d, dtype=np.float64)

                            focal_length = 1 * img_w
                            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                                 [0, focal_length, img_w / 2],
                                                 [0, 0, 1]])
                            dist_matrix = np.zeros((4, 1))

                            success, rot_vec, trans_vec = cv2.solvePnP(
                                face_3d, face_2d, cam_matrix, dist_matrix)
                            rmat, _ = cv2.Rodrigues(rot_vec)
                            angles, *_ = cv2.RQDecomp3x3(rmat)
                            x_angle, y_angle = angles[0] * 360, angles[1] * 360

                            head_pose.X_AXIS_CHEAT = 1 if y_angle < -10 or y_angle > 10 else 0
                            head_pose.Y_AXIS_CHEAT = 1 if x_angle < -5 or x_angle > 5 else 0

                            direction = "Forward"
                            if y_angle < -10:
                                direction = "Looking Left"
                            elif y_angle > 10:
                                direction = "Looking Right"
                            elif x_angle > 5:
                                direction = "Looking Up"
                            elif x_angle < -5:
                                direction = "Looking Down"

                            cv2.putText(annotated_frame, f"Primary: {direction}", (10, 90),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                head_pose.MULTIPLE_FACE_CHEAT = 0
                head_pose.X_AXIS_CHEAT = 0
                head_pose.Y_AXIS_CHEAT = 0

            # Phone Detection
            with torch.no_grad():
                results = phone_detection.model(frame)
            detections = results.pandas().xyxy[0]
            phone_detection.PHONE_CHEAT = 1 if len(detections) > 0 else 0

            for _, det in detections.iterrows():
                x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated_frame, f"Phone {det['confidence']:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Show Combined Window
            cv2.imshow("Proctoring View", annotated_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                stop_event.set()
                break

    cv2.destroyAllWindows()
