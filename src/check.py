import cv2

cap = cv2.VideoCapture(0)  # Try 0, 1, 2, etc., if multiple cameras exist

if not cap.isOpened():
    print("Error: Webcam not accessible")
else:
    print("Webcam connected")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    cv2.imshow("External Webcam", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()