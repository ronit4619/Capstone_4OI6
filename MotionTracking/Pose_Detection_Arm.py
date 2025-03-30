import cv2
import numpy as np
import mediapipe as mp
import math
from ultralytics import YOLO

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Load the YOLO model
model = YOLO('basketball_and_hoop.pt')

def calculate_angle(landmarks, arm_to_scan):
    if arm_to_scan == "left":
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    elif arm_to_scan == "right":
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    else:
        return None, None, None, None

    angle = np.arctan2(wrist[1] - elbow[1], wrist[0] - elbow[0]) - np.arctan2(shoulder[1] - elbow[1], shoulder[0] - elbow[0])
    angle = np.abs(angle * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return shoulder, elbow, wrist, int(angle)

def is_basketball_in_frame(image, wrist):
    results = model(image)
    basketball_detected = False
    for result in results:
        for box in result.boxes:
            if box.cls == 0:  # Assuming class 0 is the ball
                basketball_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(image, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

    if basketball_detected:
        cv2.putText(image, "Basketball", (350, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return basketball_detected

def main():
    while True:
        arm_to_scan = input("Which arm would you like to scan? (left/right/default): ").strip().lower()
        if arm_to_scan in ["left", "right", "default"]:
            break
        else:
            print("Invalid input. Please enter 'left', 'right', or 'default'.")

    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        prev_angle = None
        angle_threshold = 3  # Threshold for angle change
        shots = 0
        in_correct_pose = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                if arm_to_scan in ["left", "right"]:
                    shoulder, elbow, wrist, angle = calculate_angle(landmarks, arm_to_scan)
                    if shoulder and elbow and wrist:
                        cv2.putText(image, str(angle),
                                    tuple(np.multiply(elbow, [640, 480]).astype(int)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                        if prev_angle is None or abs(angle - prev_angle) >= angle_threshold:
                            prev_angle = angle

                        if 60 <= angle <= 100 and is_basketball_in_frame(image, wrist)==True:
                            angle_status = "Angle: Correct"
                            in_correct_pose = True
                        else:
                            angle_status = "Angle: Incorrect"
                            if in_correct_pose and (angle > 120):
                                shots += 1
                                in_correct_pose = False

                        cv2.putText(image, angle_status,
                                    (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if angle_status == "Angle: Correct" else (0, 0, 255), 2, cv2.LINE_AA)

                        cv2.putText(image, f"Correct Shots: {shots}",
                                    (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                        if arm_to_scan == "left":
                            left_arm_connections = [(shoulder, elbow), (elbow, wrist)]
                            for connection in left_arm_connections:
                                cv2.line(image, 
                                         tuple(np.multiply(connection[0], [640, 480]).astype(int)),
                                         tuple(np.multiply(connection[1], [640, 480]).astype(int)),
                                         (0, 255, 0), 2)
                        elif arm_to_scan == "right":
                            right_arm_connections = [(shoulder, elbow), (elbow, wrist)]
                            for connection in right_arm_connections:
                                cv2.line(image, 
                                         tuple(np.multiply(connection[0], [640, 480]).astype(int)),
                                         tuple(np.multiply(connection[1], [640, 480]).astype(int)),
                                         (0, 255, 0), 2)
                else:
                    # Default: Draw all landmarks and connections
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    
                    # Calculate and display angles for both arms
                    left_shoulder, left_elbow, left_wrist, left_angle = calculate_angle(landmarks, "left")
                    right_shoulder, right_elbow, right_wrist, right_angle = calculate_angle(landmarks, "right")

                    if left_elbow:
                        cv2.putText(image, str(left_angle),
                                    tuple(np.multiply(left_elbow, [640, 480]).astype(int)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    if right_elbow:
                        cv2.putText(image, str(right_angle),
                                    tuple(np.multiply(right_elbow, [640, 480]).astype(int)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)



            cv2.imshow('Webcam Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()