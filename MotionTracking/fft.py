import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from scipy.signal import savgol_filter

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

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
    angle_buffer = deque(maxlen=7)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
         mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_pose = pose.process(frame_rgb)
            results_hands = hands.process(frame_rgb)

            if results_pose.pose_landmarks:
                landmarks = results_pose.pose_landmarks.landmark

                shoulder = [landmarks[SHOULDER].x * frame.shape[1], landmarks[SHOULDER].y * frame.shape[0]]
                elbow = [landmarks[ELBOW].x * frame.shape[1], landmarks[ELBOW].y * frame.shape[0]]
                wrist = [landmarks[WRIST].x * frame.shape[1], landmarks[WRIST].y * frame.shape[0]]

                # Draw reference lines
                x_axis_start = (int(shoulder[0] - 100), int(shoulder[1]))
                x_axis_end = (int(shoulder[0] + 150), int(shoulder[1]))
                cv2.line(frame, x_axis_start, x_axis_end, (255, 255, 255), 2)
                cv2.line(frame, tuple(map(int, shoulder)), tuple(map(int, elbow)), (0, 255, 0), 2)
                cv2.line(frame, tuple(map(int, elbow)), tuple(map(int, wrist)), (0, 255, 0), 2)

                def calculate_angle(a, b, c):
                    a, b, c = np.array(a), np.array(b), np.array(c)
                    ba = a - b
                    bc = c - b
                    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

                def calculate_angle_x_axis(a, b):
                    vec = np.array(b) - np.array(a)
                    x_axis = np.array([1, 0])
                    cosine_angle = np.dot(vec, x_axis) / np.linalg.norm(vec)
                    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                shoulder_elbow_angle = calculate_angle_x_axis(shoulder, elbow)
                angle_buffer.append(shoulder_elbow_angle)

                if len(angle_buffer) == angle_buffer.maxlen:
                    smoothed_angle = savgol_filter(list(angle_buffer), window_length=7, polyorder=3)[-1]
                else:
                    smoothed_angle = shoulder_elbow_angle

                if elbow_angle >= 165 and (smoothed_angle <= 15 or smoothed_angle >= 165):
                    if stored_release_angle is None:
                        stored_release_angle = smoothed_angle
                        print(f"🎯 Stored Smoothed Release Angle: {stored_release_angle:.2f}°")

                cv2.putText(frame, f"{arm_choice.title()} Elbow Angle: {int(elbow_angle)}°", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Smoothed Shoulder-Elbow Angle: {int(smoothed_angle)}°", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                if stored_release_angle is not None:
                    cv2.putText(frame, f"Stored Release Angle: {int(stored_release_angle)}°", (50, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Hand tracking based on x-position, not label
            if results_hands.multi_hand_landmarks:
                selected_hand = None
                extreme_x = None

                for hand_landmarks in results_hands.multi_hand_landmarks:
                    wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x

                    if arm_choice == 'left':
                        if extreme_x is None or wrist_x < extreme_x:
                            extreme_x = wrist_x
                            selected_hand = hand_landmarks
                    else:  # 'right'
                        if extreme_x is None or wrist_x > extreme_x:
                            extreme_x = wrist_x
                            selected_hand = hand_landmarks

                if selected_hand:
                    wrist_h = selected_hand.landmark[mp_hands.HandLandmark.WRIST]
                    index_tip = selected_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                    wrist_px = (int(wrist_h.x * frame.shape[1]), int(wrist_h.y * frame.shape[0]))
                    tip_px = (int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0]))

                    cv2.circle(frame, wrist_px, 6, (255, 0, 0), -1)
                    cv2.circle(frame, tip_px, 6, (0, 255, 255), -1)
                    cv2.line(frame, wrist_px, tip_px, (0, 0, 255), 2)

                    # Compute release angle of finger relative to x-axis
                    vec = np.array(tip_px) - np.array(wrist_px)
                    x_axis = np.array([1, 0])
                    cosine_angle = np.dot(vec, x_axis) / np.linalg.norm(vec)
                    finger_release_angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

                    cv2.putText(frame, f"Finger Release Angle: {int(finger_release_angle)}°", (50, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

            cv2.imshow('Release Angle Detection', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                print("🛑 Exiting...")
                break

    cap.release()
    cv2.destroyAllWindows()

process_video()
