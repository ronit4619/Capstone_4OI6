import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

def process_video():
    cap = cv2.VideoCapture(0)  # Open webcam

    if not cap.isOpened():
        print("❌ Error: Could not access webcam.")
        return

    print("✅ Webcam opened successfully!")

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame.")
                break

            # Convert frame to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            # Check if pose was detected
            if results.pose_landmarks:
                print("✅ Pose detected!")
                landmarks = results.pose_landmarks.landmark

                # Get shoulder, elbow, and wrist positions
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1],
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]]
                
                elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * frame.shape[1],
                         landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * frame.shape[0]]
                
                wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * frame.shape[1],
                         landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * frame.shape[0]]

                print(f"📍 Shoulder: {shoulder}, Elbow: {elbow}, Wrist: {wrist}")

                # Draw X-Axis from Shoulder
                x_axis_start = (int(shoulder[0] - 100), int(shoulder[1]))  # Extend left
                x_axis_end = (int(shoulder[0] + 150), int(shoulder[1]))  # Extend right
                cv2.line(frame, x_axis_start, x_axis_end, (255, 255, 255), 2)

                # Draw shooting arm (shoulder -> elbow -> wrist)
                cv2.line(frame, (int(shoulder[0]), int(shoulder[1])), (int(elbow[0]), int(elbow[1])), (0, 255, 0), 2)
                cv2.line(frame, (int(elbow[0]), int(elbow[1])), (int(wrist[0]), int(wrist[1])), (0, 255, 0), 2)

                # Function to calculate angle
                def calculate_angle(a, b, c):
                    a = np.array(a)  # Shoulder
                    b = np.array(b)  # Elbow
                    c = np.array(c)  # Wrist
                    ba = a - b
                    bc = c - b
                    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
                    return angle

                # Compute release angle
                release_angle = calculate_angle(shoulder, elbow, wrist)

                # Display release angle
                cv2.putText(frame, f"Release Angle: {int(release_angle)}°", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            else:
                print("⚠️ No pose detected.")

            # Show output video
            cv2.imshow('Release Angle Detection', frame)

            # Press 'q' to exit
            if cv2.waitKey(10) & 0xFF == ord('q'):
                print("🛑 Exiting...")
                break

    cap.release()
    cv2.destroyAllWindows()

# Run the video processing
process_video()
