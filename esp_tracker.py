import cv2
import mediapipe as mp
import time

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Webcam feed
cap = cv2.VideoCapture(0)
p_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:
        # Draw skeleton
        mp_drawing.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)
        )

        # Extract landmark positions
        landmarks = result.pose_landmarks.landmark
        x_coords = [int(lm.x * w) for lm in landmarks]
        y_coords = [int(lm.y * h) for lm in landmarks]

        # Bounding box
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        cv2.rectangle(frame, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 255), 2)

        # Head label (like name tag)
        cv2.putText(frame, "TARGET", (x_min, y_min - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Health bar (left side of box)
        health_bar_height = 100
        current_health = 80  # Simulated value, could be dynamic later
        bar_x = x_min - 30
        bar_y = y_min
        filled_height = int((current_health / 100) * health_bar_height)

        # Draw health bar background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 10, bar_y + health_bar_height), (50, 50, 50), -1)
        # Draw filled health
        cv2.rectangle(frame, (bar_x, bar_y + (health_bar_height - filled_height)),
                      (bar_x + 10, bar_y + health_bar_height), (0, 255, 0), -1)

    # FPS display
    c_time = time.time()
    fps = 1 / (c_time - p_time) if p_time != 0 else 0
    p_time = c_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("REAL-LIFE ESP TRACKER", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
