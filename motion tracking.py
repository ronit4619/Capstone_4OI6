import cv2
import math
import time
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import cvzone

# ------------------- 1) Custom Pose Detector -------------------
class poseDetector():
    """
    Custom class to detect human body landmarks using MediaPipe Pose
    (wrapped for easier usage and potential reusability).
    """
    def __init__(self, 
                 static_image_mode=False,
                 model_complexity=1,
                 smooth_landmarks=True,
                 enable_segmentation=False,
                 smooth_segmentation=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.static_image_mode,
            model_complexity=self.model_complexity,
            smooth_landmarks=self.smooth_landmarks,
            enable_segmentation=self.enable_segmentation,
            smooth_segmentation=self.smooth_segmentation,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.results = None

    def findPose(self, img, draw=True):
        """Runs MediaPipe Pose on the image and optionally draws landmarks."""
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def getPosition(self, img, draw=True):
        """
        Returns a list of landmarks in the format [id, x, y].
        - id: landmark index
        - x, y: pixel coordinates in the image
        """
        lmList = []
        if self.results and self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
        return lmList

# ------------------- 2) Utility Functions -------------------
def calculate_angle(p1, p2, p3):
    """Calculates the angle (in degrees) between three points p1, p2, p3."""
    a = (p1[0] - p2[0], p1[1] - p2[1])
    b = (p3[0] - p2[0], p3[1] - p2[1])
    dot_product = a[0] * b[0] + a[1] * b[1]
    mag_a = math.sqrt(a[0] ** 2 + a[1] ** 2)
    mag_b = math.sqrt(b[0] ** 2 + b[1] ** 2)
    if mag_a == 0 or mag_b == 0:
        return 0
    angle = math.acos(dot_product / (mag_a * mag_b))
    return math.degrees(angle)

def calculate_speed(p1, p2, time_elapsed):
    """Calculates the speed of movement between two points (px/sec)."""
    if p1 is None or p2 is None or time_elapsed == 0:
        return 0
    distance = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    return distance / time_elapsed

# ------------------- 3) Shot Analysis Functions -------------------
def check_set_position(shoulder, elbow, wrist):
    """Checks if the player is in the correct set position (cocked wrist)."""
    elbow_angle = calculate_angle(shoulder, elbow, wrist)
    return 80 <= elbow_angle <= 130  # Adjust as needed

def is_one_motion(wrist_coords, prev_wrist_coords):
    """Checks if the motion follows a single path without backward looping."""
    if prev_wrist_coords is None:
        return True
    return wrist_coords[1] <= prev_wrist_coords[1]  # wrist moving upward

def check_release_point(shoulder, elbow, wrist):
    """Verifies that the arm is at the release point (~45-degree angle)."""
    release_angle = calculate_angle(shoulder, elbow, wrist)
    return 30 <= release_angle <= 60  # Adjust as needed

def analyze_shot(lmList, prev_wrist, prev_time, frame):
    """
    Analyzes the shot using the extracted 2D landmarks from the poseDetector.
    Returns: (set_position, one_motion, release_point, speed, current_wrist, current_time, arm_angle)
    """
    if len(lmList) < 16:
        # Not enough landmarks for left shoulder/elbow/wrist
        return False, False, False, 0.0, prev_wrist, time.time(), 0.0

    # MediaPipe indices: 11->Left Shoulder, 13->Left Elbow, 15->Left Wrist
    left_shoulder = (lmList[11][1], lmList[11][2])
    left_elbow = (lmList[13][1], lmList[13][2])
    left_wrist = (lmList[15][1], lmList[15][2])

    set_position = check_set_position(left_shoulder, left_elbow, left_wrist)
    one_motion_flag = is_one_motion(left_wrist, prev_wrist)
    release_point = check_release_point(left_shoulder, left_elbow, left_wrist)

    current_time = time.time()
    time_elapsed = current_time - prev_time
    speed = calculate_speed(prev_wrist, left_wrist, time_elapsed)

    # Calculate arm angle at the elbow
    arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

    return set_position, one_motion_flag, release_point, speed, left_wrist, current_time, arm_angle

# ------------------- 4) YOLO Basketball Detection -------------------
def detect_basketball(model, frame):
    """Detects the basketball in the frame using the YOLO model."""
    results = model(frame)
    for result in results:
        for box in result.boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
            return (x_min, y_min, x_max, y_max)
    return None

# ------------------- 5) Main Program with Mode Selection -------------------
def main():
    """
    Main live camera loop that supports two modes:
      - Default Mode: Only player & ball detection (no ball tracking)
      - Ball Tracking Mode: Toggled with 'S'. In this mode, the ball is detected using YOLO,
        and its center is stored to track the ball's path. Polynomial regression is applied to draw
        the trajectory, and a simple basket prediction is made.
        * Press 'R' (while in ball tracking mode) to purge the existing tracking data.
        * Press 'S' again to toggle ball tracking mode off.
    """
    # 1) Load your YOLO model
    model_path = r"C:\Users\biswa\Downloads\best.pt"  # Update this path as needed
    basketball_model = YOLO(model_path)

    # 2) Initialize your custom pose detector
    detector = poseDetector(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # 3) Initialize lists to store ball tracking positions (for polynomial regression)
    posListX, posListY = [], []
    prediction = False

    # --- Mode flag ---
    ball_tracking_mode = False     # Default: OFF (no ball tracking)

    # 4) Open the Webcam (live usage)
    cap = cv2.VideoCapture(0)
    prev_wrist = None
    prev_time = time.time()
    
    # Create a named window and set it to full-screen mode
    window_name = "Basketball Detection and Shooting Form Analysis"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video.")
            break

        # ------------------ Pose Detection (always active) ------------------
        frame = detector.findPose(frame, draw=True)
        lmList = detector.getPosition(frame, draw=False)

        # ------------------ YOLO: Detect Basketball (always active) ------------------
        basketball_box = detect_basketball(basketball_model, frame)
        if basketball_box:
            x_min, y_min, x_max, y_max = basketball_box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, "Basketball", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # ------------------ Shooting Form Analysis (Pose) ------------------
        arm_angle = 0.0  # Default value if no landmarks are detected
        if len(lmList) >= 16:
            set_ok, one_motion_flag, release_ok, speed, prev_wrist, prev_time, arm_angle = analyze_shot(
                lmList, prev_wrist, prev_time, frame
            )

            if set_ok:
                cv2.putText(frame, "Set Position: OK", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Set Position: Adjust your shoulder/elbow", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            if one_motion_flag:
                cv2.putText(frame, "Motion: Smooth", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Motion: Improve your release", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            if release_ok:
                cv2.putText(frame, "Release: Good!", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Release: Adjust angle", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.putText(frame, f"Wrist Speed: {speed:.2f} px/sec", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Always display the arm angle
        cv2.putText(frame, f"Arm Angle: {arm_angle:.2f} degrees", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # ------------------ Ball Tracking (only if ball_tracking_mode is ON) ------------------
        if ball_tracking_mode:
            ball_box_for_tracking = detect_basketball(basketball_model, frame)
            center_yolo = None
            if ball_box_for_tracking:
                x_min, y_min, x_max, y_max = ball_box_for_tracking
                center_yolo = (int((x_min + x_max) / 2), int((y_min + y_max) / 2))
                cv2.circle(frame, center_yolo, 5, (0, 0, 255), -1)
            
            if center_yolo is not None:
                posListX.append(center_yolo[0])
                posListY.append(center_yolo[1])
            
            if len(posListX) >= 3:
                A, B, C = np.polyfit(posListX, posListY, 2)
                for i, (posX, posY) in enumerate(zip(posListX, posListY)):
                    cv2.circle(frame, (posX, posY), 10, (0, 255, 0), cv2.FILLED)
                    if i > 0:
                        cv2.line(frame, (posListX[i - 1], posListY[i - 1]), (posX, posY), (0, 255, 0), 2)
                
                for x in range(0, frame.shape[1], 10):
                    y = int(A * (x ** 2) + B * x + C)
                    cv2.circle(frame, (x, y), 2, (255, 0, 255), cv2.FILLED)
                
                a, b, c = A, B, C - 590
                discriminant = b ** 2 - 4 * a * c
                if discriminant >= 0:
                    x_pred = int((-b - math.sqrt(discriminant)) / (2 * a))
                    prediction = 330 < x_pred < 430
                else:
                    prediction = False

                if prediction:
                    cvzone.putTextRect(frame, "Basket", (50, 150), scale=3,
                                       thickness=3, colorR=(0, 200, 0), offset=20)
                else:
                    cvzone.putTextRect(frame, "No Basket", (50, 150), scale=3,
                                       thickness=3, colorR=(0, 0, 200), offset=20)
        # ------------------ End Ball Tracking ------------------
       
        # Determine which mode text to display
        if ball_tracking_mode:
            mode_text = "Ball Tracking Mode"
        else:
            mode_text = "Default Mode"

        (text_width, text_height), _ = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        frame_width = frame.shape[1]
        text_x = frame_width - text_width - 10
        text_y = text_height + 10

        cv2.putText(frame, mode_text, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow(window_name, frame)

        # --------------- Key handling ---------------
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            ball_tracking_mode = not ball_tracking_mode
            posListX.clear()
            posListY.clear()
            print("Ball tracking mode ENABLED." if ball_tracking_mode else "Ball tracking mode DISABLED.")
        elif key == ord('r'):
            if ball_tracking_mode:
                posListX.clear()
                posListY.clear()
                print("Ball tracking data purged. Restarting ball tracking.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
