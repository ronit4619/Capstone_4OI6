import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from scipy.signal import savgol_filter

mp_pose = mp.solutions.pose

def process_video():
    arm_choice = input("👉 Which arm would you like to track? Type 'left' or 'right': ").strip().lower()
    if arm_choice not in ['left', 'right']:
        print("❌ Invalid choice. Please enter 'left' or 'right'.")
        return

    if arm_choice == 'left':
        SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER.value
        ELBOW = mp_pose.PoseLandmark.LEFT_ELBOW.value
        WRIST = mp_pose.PoseLandmark.LEFT_WRIST.value
    else:
        SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        ELBOW = mp_pose.PoseLandmark.RIGHT_ELBOW.value
        WRIST = mp_pose.PoseLandmark.RIGHT_WRIST.value

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Could not access webcam.")
        return

    print("✅ Webcam opened successfully!")

    stored_release_angle = None

    # Initialize buffer for smoothing
    angle_buffer = deque(maxlen=7)  # Window length = 5

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                shoulder = [landmarks[SHOULDER].x * frame.shape[1],
                            landmarks[SHOULDER].y * frame.shape[0]]
                elbow = [landmarks[ELBOW].x * frame.shape[1],
                         landmarks[ELBOW].y * frame.shape[0]]
                wrist = [landmarks[WRIST].x * frame.shape[1],
                         landmarks[WRIST].y * frame.shape[0]]

                # Draw X-axis reference line
                x_axis_start = (int(shoulder[0] - 100), int(shoulder[1]))
                x_axis_end = (int(shoulder[0] + 150), int(shoulder[1]))
                cv2.line(frame, x_axis_start, x_axis_end, (255, 255, 255), 2)

                cv2.line(frame, (int(shoulder[0]), int(shoulder[1])), (int(elbow[0]), int(elbow[1])), (0, 255, 0), 2)
                cv2.line(frame, (int(elbow[0]), int(elbow[1])), (int(wrist[0]), int(wrist[1])), (0, 255, 0), 2)

                def calculate_angle(a, b, c):
                    a = np.array(a)
                    b = np.array(b)
                    c = np.array(c)
                    ba = a - b
                    bc = c - b
                    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
                    return angle

                def calculate_angle_x_axis(shoulder, elbow):
                    vec = np.array(elbow) - np.array(shoulder)
                    x_axis = np.array([1, 0])
                    cosine_angle = np.dot(vec, x_axis) / np.linalg.norm(vec)
                    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
                    return angle

                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                shoulder_elbow_angle = calculate_angle_x_axis(shoulder, elbow)

                # Append raw shoulder_elbow_angle to buffer
                angle_buffer.append(shoulder_elbow_angle)

                # Apply Savgol filter if buffer is full
                if len(angle_buffer) == angle_buffer.maxlen:
                    smoothed_angle = savgol_filter(list(angle_buffer), window_length=7, polyorder=3)[-1] # play around with the parameters 
                else:
                    smoothed_angle = shoulder_elbow_angle

                if elbow_angle >= 165 and (smoothed_angle <= 15 or smoothed_angle >= 165):
                    if stored_release_angle is None:
                        stored_release_angle = smoothed_angle
                        print(f"🎯 Stored Smoothed Release Angle: {stored_release_angle:.2f}°")

                # Display info
                cv2.putText(frame, f"{arm_choice.title()} Elbow Angle: {int(elbow_angle)}°", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Smoothed Shoulder-Elbow Angle: {int(smoothed_angle)}°", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                if stored_release_angle is not None:
                    cv2.putText(frame, f"Stored Release Angle: {int(stored_release_angle)}°", (50, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                print("⚠️ No pose detected.")

            cv2.imshow('Release Angle Detection', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                print("🛑 Exiting...")
                break

    cap.release()
    cv2.destroyAllWindows()

process_video()
