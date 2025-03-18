import cv2
import numpy as np
import mediapipe as mp
import math
from ultralytics import YOLO

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Load the YOLO model
model = YOLO('basketball_and_hoop.pt')

def calculate_angle(landmarks, limb_to_scan):
    if limb_to_scan == "left_arm":
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        angle = np.arctan2(wrist[1] - elbow[1], wrist[0] - elbow[0]) - np.arctan2(shoulder[1] - elbow[1], shoulder[0] - elbow[0])
    elif limb_to_scan == "right_arm":
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        angle = np.arctan2(wrist[1] - elbow[1], wrist[0] - elbow[0]) - np.arctan2(shoulder[1] - elbow[1], shoulder[0] - elbow[0])
    elif limb_to_scan == "left_leg":
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
               landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        angle = np.arctan2(ankle[1] - knee[1], ankle[0] - knee[0]) - np.arctan2(hip[1] - knee[1], hip[0] - knee[0])
    elif limb_to_scan == "right_leg":
        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
               landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        angle = np.arctan2(ankle[1] - knee[1], ankle[0] - knee[0]) - np.arctan2(hip[1] - knee[1], hip[0] - knee[0])
    else:
        return None, None, None, None

    angle = np.abs(angle * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle

    return shoulder if 'shoulder' in locals() else hip, elbow if 'elbow' in locals() else knee, wrist if 'wrist' in locals() else ankle, int(angle)

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

def is_landmark_visible(landmark):
    return 0 <= landmark.x <= 1 and 0 <= landmark.y <= 1

def draw_text_with_background(image, text, position, font, font_scale, font_color, thickness, bg_color):
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size
    x, y = position
    cv2.rectangle(image, (x, y - text_h - 10), (x + text_w, y + 10), bg_color, -1)
    cv2.putText(image, text, (x, y), font, font_scale, font_color, thickness)

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
                    shoulder, elbow, wrist, angle = calculate_angle(landmarks, f"{arm_to_scan}_arm")
                    hip, knee, ankle, leg_angle = calculate_angle(landmarks, f"{arm_to_scan}_leg")

                    missing_landmarks = []
                    if not is_landmark_visible(landmarks[mp_pose.PoseLandmark[f"{arm_to_scan.upper()}_SHOULDER"].value]):
                        missing_landmarks.append("Shoulder")
                    if not is_landmark_visible(landmarks[mp_pose.PoseLandmark[f"{arm_to_scan.upper()}_ELBOW"].value]):
                        missing_landmarks.append("Elbow")
                    if not is_landmark_visible(landmarks[mp_pose.PoseLandmark[f"{arm_to_scan.upper()}_WRIST"].value]):
                        missing_landmarks.append("Wrist")
                    if not is_landmark_visible(landmarks[mp_pose.PoseLandmark[f"{arm_to_scan.upper()}_HIP"].value]):
                        missing_landmarks.append("Hip")
                    if not is_landmark_visible(landmarks[mp_pose.PoseLandmark[f"{arm_to_scan.upper()}_KNEE"].value]):
                        missing_landmarks.append("Knee")
                    if not is_landmark_visible(landmarks[mp_pose.PoseLandmark[f"{arm_to_scan.upper()}_ANKLE"].value]):
                        missing_landmarks.append("Ankle")

                    if missing_landmarks:
                        draw_text_with_background(image, f"Missing: {', '.join(missing_landmarks)}",
                                                  (50, image.shape[0] - 50),  # Position at the bottom of the screen
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, (0, 0, 0))  # Smaller font size
                    else:
                        if shoulder and elbow and wrist:
                            draw_text_with_background(image, str(angle),
                                                      tuple(np.multiply(elbow, [640, 480]).astype(int)),
                                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, (0, 0, 0))
                        if hip and knee and ankle:
                            draw_text_with_background(image, str(leg_angle),
                                                      tuple(np.multiply(knee, [640, 480]).astype(int)),
                                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, (0, 0, 0))

                        if prev_angle is None or abs(angle - prev_angle) >= angle_threshold:
                            prev_angle = angle

                        if (60 <= angle <= 100) and (85 <= leg_angle <= 155) and is_basketball_in_frame(image, wrist):
                            angle_status = "Pose: Correct"
                            in_correct_pose = True
                        else:
                            angle_status = "Pose: Incorrect"
                            if in_correct_pose and (angle > 135):  # replace angle >135 to (is_basketball_in_frame(image, wrist)==False)
                                shots += 1
                                in_correct_pose = False

                        draw_text_with_background(image, angle_status,
                                                  (50, 50),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if angle_status == "Pose: Correct" else (0, 0, 255), 2, (0, 0, 0))

                        draw_text_with_background(image, f"Correct Shots: {shots}",
                                                  (50, 100),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, (0, 0, 0))

                        if arm_to_scan == "left":
                            left_arm_connections = [(shoulder, elbow), (elbow, wrist)]
                            left_leg_connections = [(hip, knee), (knee, ankle)]
                            for connection in left_arm_connections + left_leg_connections:
                                cv2.line(image,
                                         tuple(np.multiply(connection[0], [640, 480]).astype(int)),
                                         tuple(np.multiply(connection[1], [640, 480]).astype(int)),
                                         (0, 255, 0), 2)
                        elif arm_to_scan == "right":
                            right_arm_connections = [(shoulder, elbow), (elbow, wrist)]
                            right_leg_connections = [(hip, knee), (knee, ankle)]
                            for connection in right_arm_connections + right_leg_connections:
                                cv2.line(image,
                                         tuple(np.multiply(connection[0], [640, 480]).astype(int)),
                                         tuple(np.multiply(connection[1], [640, 480]).astype(int)),
                                         (0, 255, 0), 2)
                else:
                    # Default: Draw all landmarks and connections
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    
                    # Calculate and display angles for both arms and legs
                    left_shoulder, left_elbow, left_wrist, left_angle = calculate_angle(landmarks, "left_arm")
                    right_shoulder, right_elbow, right_wrist, right_angle = calculate_angle(landmarks, "right_arm")
                    left_hip, left_knee, left_ankle, left_leg_angle = calculate_angle(landmarks, "left_leg")
                    right_hip, right_knee, right_ankle, right_leg_angle = calculate_angle(landmarks, "right_leg")

                    if left_elbow:
                        draw_text_with_background(image, str(left_angle),
                                                  tuple(np.multiply(left_elbow, [640, 480]).astype(int)),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, (0, 0, 0))
                    if right_elbow:
                        draw_text_with_background(image, str(right_angle),
                                                  tuple(np.multiply(right_elbow, [640, 480]).astype(int)),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, (0, 0, 0))
                    if left_knee:
                        draw_text_with_background(image, str(left_leg_angle),
                                                  tuple(np.multiply(left_knee, [640, 480]).astype(int)),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, (0, 0, 0))
                    if right_knee:
                        draw_text_with_background(image, str(right_leg_angle),
                                                  tuple(np.multiply(right_knee, [640, 480]).astype(int)),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, (0, 0, 0))

            cv2.imshow('Webcam Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
