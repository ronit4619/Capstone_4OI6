import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from scipy.signal import savgol_filter
from ultralytics import YOLO
import time

mp_pose = mp.solutions.pose
model = YOLO('best.pt')

# Initialize tracker state
tracker = None
tracking_ball = False
lost_tracker_frames = 0
tracker_lost_threshold = 10
last_yolo_check_time = 0
recheck_interval = 0.3  # seconds #steph curry release is 0.4

def detect_and_track_basketball(image, force_redetect=False):
    global tracker, tracking_ball, lost_tracker_frames, last_yolo_check_time

    current_time = time.time()
    if force_redetect:
        tracking_ball = False
        tracker = None
        print("🔄 Manual reset: Forcing re-detection.")

    # Periodic YOLO re-check every recheck_interval seconds
    if tracking_ball and (current_time - last_yolo_check_time > recheck_interval):
        print("🔁 Performing periodic YOLO confirmation.")
        tracking_ball = False
        tracker = None

    if tracking_ball and tracker is not None:
        success, box = tracker.update(image)
        if success:
            lost_tracker_frames = 0
            x, y, w, h = map(int, box)
            cx, cy = x + w // 2, y + h // 2
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, "Tracking", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
            return True, (cx, cy)
        else:
            lost_tracker_frames += 1
            if lost_tracker_frames >= tracker_lost_threshold:
                print("🛑 Tracker lost the ball. Switching to re-detection.")
                tracking_ball = False
                tracker = None
            return False, None

    # If not tracking or manually forced to re-detect
    results = model(image)
    last_yolo_check_time = current_time  # reset timer on detection
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            if class_id == 0 and confidence > 0.80:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                tracker = cv2.TrackerCSRT_create()
                tracker.init(image, (x1, y1, w, h))
                tracking_ball = True
                lost_tracker_frames = 0
                cx, cy = x1 + w // 2, y1 + h // 2
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"basketball: {confidence:.2f}"
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                return True, (cx, cy)

    return False, None


def process_video():
    arm_choice = input("👉 Which arm would you like to track? Type 'left' or 'right': ").strip().lower()
    if arm_choice not in ['left', 'right']:
        print("❌ Invalid choice. Please enter 'left' or 'right'.")
        return

    SHOULDER = getattr(mp_pose.PoseLandmark, f"{arm_choice.upper()}_SHOULDER").value
    ELBOW = getattr(mp_pose.PoseLandmark, f"{arm_choice.upper()}_ELBOW").value
    WRIST = getattr(mp_pose.PoseLandmark, f"{arm_choice.upper()}_WRIST").value

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not access webcam.")
        return

    print("✅ Webcam opened successfully!")
    print("🎮 Press 'R' to re-detect basketball | 'Q' to quit")
    stored_release_angle = None
    angle_buffer = deque(maxlen=7)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            key = cv2.waitKey(1) & 0xFF
            force_redetect = (key == ord('r'))

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                shoulder = [landmarks[SHOULDER].x * frame.shape[1], landmarks[SHOULDER].y * frame.shape[0]]
                elbow = [landmarks[ELBOW].x * frame.shape[1], landmarks[ELBOW].y * frame.shape[0]]
                wrist = [landmarks[WRIST].x * frame.shape[1], landmarks[WRIST].y * frame.shape[0]]

                x_axis_start = (int(shoulder[0] - 100), int(shoulder[1]))
                x_axis_end = (int(shoulder[0] + 150), int(shoulder[1]))
                cv2.line(frame, x_axis_start, x_axis_end, (255, 255, 255), 2)
                cv2.line(frame, tuple(map(int, shoulder)), tuple(map(int, elbow)), (0, 255, 0), 2)
                cv2.line(frame, tuple(map(int, elbow)), tuple(map(int, wrist)), (0, 255, 0), 2)

                def calculate_angle(a, b, c):
                    a, b, c = map(np.array, (a, b, c))
                    ba = a - b
                    bc = c - b
                    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

                def calculate_angle_x_axis(shoulder, elbow):
                    vec = np.array(elbow) - np.array(shoulder)
                    x_axis = np.array([1, 0])
                    cosine_angle = np.dot(vec, x_axis) / np.linalg.norm(vec)
                    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                shoulder_elbow_angle = calculate_angle_x_axis(shoulder, elbow)
                angle_buffer.append(shoulder_elbow_angle)

                smoothed_angle = savgol_filter(list(angle_buffer), window_length=7, polyorder=3)[-1] \
                    if len(angle_buffer) == angle_buffer.maxlen else shoulder_elbow_angle

                if elbow_angle >= 165 and (smoothed_angle <= 15 or smoothed_angle >= 165):
                    if stored_release_angle is None:
                        stored_release_angle = smoothed_angle
                        print(f"🎯 Stored Smoothed Release Angle: {stored_release_angle:.2f}°")

                ball_detected, ball_center = detect_and_track_basketball(frame, force_redetect)

                if ball_detected:
                    cv2.putText(frame, "🎯 Basketball Detected!", (50, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                elif not tracking_ball:
                    cv2.putText(frame, "👋 Press 'R' to detect basketball", (50, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                cv2.putText(frame, f"{arm_choice.title()} Elbow Angle: {int(elbow_angle)}°", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Smoothed Shoulder-Elbow Angle: {int(smoothed_angle)}°", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                if stored_release_angle is not None:
                    cv2.putText(frame, f"Stored Release Angle: {int(stored_release_angle)}°", (50, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "🕵️ Waiting for pose detection...", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.putText(frame, "Press 'R' to re-detect | 'Q' to quit", (50, frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow('Release Angle Detection', frame)

            if key == ord('q'):
                print("🛑 Exiting...")
                break

    cap.release()
    cv2.destroyAllWindows()

process_video()
