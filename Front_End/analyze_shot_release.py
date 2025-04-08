#analyze_shot_release.py
# This script detects the release angle of a basketball shot using OpenCV and MediaPipe Pose.
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from scipy.signal import savgol_filter
from ultralytics import YOLO
import time
import torch

import json
from datetime import datetime


from flask import Flask, Response, request
import argparse

# Argument parser to read arm and save options
parser = argparse.ArgumentParser()
parser.add_argument('--arm', type=str, default="default")  # default: use both arms
args = parser.parse_args()

# Use parsed arguments instead of input()
arm_to_scan = args.arm.lower()

# Create a Flask app
app = Flask(__name__)


torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'


mp_pose = mp.solutions.pose
model = YOLO('best.pt')
model.to(torch_device)

model_person = YOLO('yolov8n.pt')
model_person.to(torch_device)

# Initialize tracker state global variables 
tracker = None
tracking_ball = False
lost_tracker_frames = 0
tracker_lost_threshold = 10
last_yolo_check_time = 0
recheck_interval = 0.1  # seconds #steph curry release is 0.4
last_player_check_time = 0
player_recheck_interval = 0.1  # in seconds


player_tracker = None
tracking_player = False
player_tracker_box = None
player_lost_frames = 0
player_lost_threshold = 10




import numpy as np


# def store_release_angle_if_valid(frame, smoothed_angle, ball_center, wrist_point, stored_release_angle, threshold=90):
#     """
#     Stores the release angle if the ball is far enough from the wrist and pose is valid.

#     Args:
#         frame (ndarray): Current video frame for annotation.
#         smoothed_angle (float): Smoothed shoulder-elbow angle.
#         ball_center (tuple): (x, y) position of the basketball.
#         wrist_point (tuple): (x, y) position of the wrist.
#         stored_release_angle (float or None): Currently stored release angle.
#         threshold (int): Minimum pixel distance between ball and wrist to count as release.

#     Returns:
#         float: Updated stored release angle.
#     """
#     if ball_center is None or wrist_point is None:
#         return stored_release_angle

#     distance = calculate_distance(ball_center, wrist_point)

#     if distance >= threshold:
#         stored_release_angle = smoothed_angle
#         cv2.putText(frame, f"Release Angle: {int(stored_release_angle)}°", 
#                     (50, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

#     # Draw line and distance regardless
#     cv2.putText(frame, f"Ball-Wrist Dist: {int(distance)} px", 
#                 (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

#     return stored_release_angle
def log_release_angle_to_json(release_angle, filename="release_angles.json"):
    data = []

    # Try loading existing data
    try:
        with open(filename, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass  # File doesn't exist yet or is empty

    # Add new release
    data.append({
        "timestamp": datetime.now().isoformat(),
        "release_angle": round(release_angle, 2)
    })

    # Write updated list back to file
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def calculate_distance(point1, point2):
    """
    Calculates the Euclidean distance between two (x, y) points.
    
    Args:
        point1: Tuple (x1, y1)
        point2: Tuple (x2, y2)
    
    Returns:
        Float distance between the two points
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))

# def is_basketball_supported_by_wrist_directional(shoulder, elbow, wrist, ball_center, distance_threshold=90, angle_threshold=90):
#     """
#     Determines if the basketball is near the wrist and in the direction of the forearm (elbow to wrist).

#     Args:
#         shoulder, elbow, wrist (tuple): Coordinates of the joint landmarks.
#         ball_center (tuple): Coordinates of the basketball center.
#         distance_threshold (float): Max distance between wrist and ball.
#         angle_threshold (float): Max angle difference between arm direction and ball vector.

#     Returns:
#         bool: True if ball is near wrist and in shooting direction.
#     """
#     if wrist is None or ball_center is None or elbow is None:
#         return False

#     wrist = np.array(wrist)
#     elbow = np.array(elbow)
#     ball_center = np.array(ball_center)

#     arm_direction = wrist - elbow
#     ball_vector = ball_center - wrist

#     distance = np.linalg.norm(ball_vector)
#     if distance > distance_threshold:
#         return False

#     # Normalize vectors to get angle
#     arm_dir_norm = arm_direction / np.linalg.norm(arm_direction)
#     ball_vec_norm = ball_vector / np.linalg.norm(ball_vector)

#     dot_product = np.dot(arm_dir_norm, ball_vec_norm)
#     angle = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))

#     return angle < angle_threshold

def is_ball_near_wrist(wrist, ball_center, threshold=100):
    """
    Returns True if the ball is within a certain pixel distance of the wrist.

    Args:
        wrist (tuple): (x, y) coordinates of the wrist.
        ball_center (tuple): (x, y) coordinates of the basketball.
        threshold (int): Distance threshold in pixels.

    Returns:
        bool: True if ball is within threshold distance from the wrist.
    """
    if wrist is None or ball_center is None:
        return False

    distance = calculate_distance(wrist, ball_center)
    return distance < threshold


# def store_release_angle_if_valid(frame, smoothed_angle, shoulder, elbow, wrist, ball_center, stored_release_angle, distance_threshold=150):
#     """
#     Stores the release angle if the ball was first supported by the wrist and then moved away.

#     Args:
#         frame (ndarray): Current video frame for annotation.
#         smoothed_angle (float): Smoothed shoulder-elbow angle.
#         shoulder, elbow, wrist (tuple): Joint coordinates.
#         ball_center (tuple): Basketball center.
#         stored_release_angle (float or None): Currently stored release angle.
#         distance_threshold (int): Distance to determine shot release.

#     Returns:
#         float: Updated stored release angle.
#     """
#     if None in [shoulder, elbow, wrist, ball_center]:
#         return stored_release_angle

#     # Check if the ball was previously supported
#     if is_ball_near_wrist(wrist, ball_center, threshold=100):
#         # Now check if it's far enough to be considered a shot
#         distance = calculate_distance(ball_center, wrist)
#         if distance >= distance_threshold:
#             stored_release_angle = smoothed_angle
#             if stored_release_angle is not None:
#                            log_release_angle_to_json(stored_release_angle)
#             cv2.putText(frame, f"Release Angle: {int(stored_release_angle)}°", 
#                         (50, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

#     # Draw distance always for feedback
#     distance = calculate_distance(ball_center, wrist)
#     cv2.putText(frame, f"Ball-Wrist Dist: {int(distance)} px", 
#                 (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

#     return stored_release_angle


def store_release_angle_if_valid(frame, smoothed_angle, wrist, ball_center,
                                  stored_release_angle, distance_threshold=150,
                                  ball_was_near_wrist=False):
    """
    Stores the release angle if the ball was previously near the wrist and has now moved away.

    Args:
        frame (ndarray): Current video frame for annotation.
        smoothed_angle (float): Smoothed shoulder-elbow angle.
        wrist (tuple): Wrist coordinates.
        ball_center (tuple): Ball coordinates.
        stored_release_angle (float or None): Previously stored release angle.
        distance_threshold (int): Distance in px to count as 'released'.
        ball_was_near_wrist (bool): Was the ball near the wrist previously?

    Returns:
        tuple: (updated_release_angle, updated_ball_was_near_wrist)
    """
    if None in [wrist, ball_center]:
        return stored_release_angle, ball_was_near_wrist

    distance = calculate_distance(ball_center, wrist)

    # Step 1: Detect "ball near wrist"
    if distance < 100:
        ball_was_near_wrist = True

    # Step 2: Detect "ball moved away" after being near
    elif ball_was_near_wrist and distance >= distance_threshold:
        stored_release_angle = smoothed_angle
        ball_was_near_wrist = False  # reset state
        if stored_release_angle is not None:
            log_release_angle_to_json(stored_release_angle)
        cv2.putText(frame, f"Release Angle: {int(stored_release_angle)}°",
                    (50, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Always show distance
    cv2.putText(frame, f"Ball-Wrist Dist: {int(distance)} px",
                (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    return stored_release_angle, ball_was_near_wrist


# def is_valid_shooting_pose_fsm(smoothed_angle, state):
#     """
#     Implements a finite state machine to detect a valid shooting pose transition.
#     Expects angle to go: ~50° ➝ ~0° ➝ ~50°

#     Args:
#         smoothed_angle (float): Shoulder-elbow smoothed angle.
#         state (str): Current FSM state.

#     Returns:
#         new_state (str): Updated FSM state.
#         is_release_frame (bool): True only when a valid release motion is detected.
#     """
#     is_release_frame = False

#     if state == 'ready' and smoothed_angle >= 50:
#         state = 'releasing'
#     elif state == 'releasing' and smoothed_angle >= 20:
#         state = 'recovered'
#         is_release_frame = True
#     elif state == 'recovered':
#         # Reset FSM to allow another detection
#         state = 'ready'

#     return state, is_release_frame

def get_closest_player_to_ball(ball_center, player_boxes):
    """
    Finds the player closest to the basketball.

    Args:
        ball_center (tuple): (x, y) coordinates of the ball.
        player_boxes (list): List of (x1, y1, x2, y2, confidence) for each detected player.

    Returns:
        tuple or None: (x1, y1, x2, y2) of the closest player, or None if no players.
    """
    if ball_center is None or not player_boxes:
        return None

    min_distance = float('inf')
    closest_box = None

    for (x1, y1, x2, y2, _) in player_boxes:
        player_center = get_box_center(x1, y1, x2, y2)
        dist = calculate_distance(ball_center, player_center)

        if dist < min_distance:
            min_distance = dist
            closest_box = (x1, y1, x2, y2)

    return closest_box


def is_valid_shooting_pose(angle): # have to fix the angle
    """
    Determines whether the user's pose is correct and the basketball is in frame.

    Args:
        angle (float): Arm angle.
        leg_angle (float): Leg angle.
        image (ndarray): Current video frame.
        wrist_point (tuple or list): (x, y) coordinates of the wrist.

    Returns:
        bool: True if pose is valid and ball is visible, False otherwise.
    """
    arm_ok = 0 <= angle <= 10
    
    #ball_ok = ballDetected

    return arm_ok


def get_box_center(x1, y1, x2, y2):
    """
    Returns the center (cx, cy) of a bounding box.
    """
    cx = x1 + (x2 - x1) // 2
    cy = y1 + (y2 - y1) // 2
    return (cx, cy)

def detect_and_track_basketball_and_players(image, force_redetect=False):
    global tracker, tracking_ball, lost_tracker_frames, last_yolo_check_time
    global last_player_check_time
    global player_tracker, tracking_player, player_tracker_box, player_lost_frames

    current_time = time.time()
    player_boxes = []
    ball_center = None

    if force_redetect:
        tracking_ball = False
        tracker = None
        tracking_player = False
        player_tracker = None
        print("🔄 Manual reset: Forcing re-detection.")

    # Ball YOLO detection every interval
    if current_time - last_yolo_check_time > recheck_interval:
        tracking_ball = False
        tracker = None

    # Ball tracking
    if tracking_ball and tracker is not None:
        success, box = tracker.update(image)
        if success:
            lost_tracker_frames = 0
            x, y, w, h = map(int, box)
            cx, cy = get_box_center(x, y, x + w, y + h)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, "Tracking Ball", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
            ball_center = (cx, cy)
        else:
            lost_tracker_frames += 1
            if lost_tracker_frames >= tracker_lost_threshold:
                print("🛑 Tracker lost the ball.")
                tracking_ball = False
                tracker = None

    if not tracking_ball:
        ball_results = model(image)
        last_yolo_check_time = current_time
        for result in ball_results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if class_id == 0 and confidence > 0.70:
                    w, h = x2 - x1, y2 - y1
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(image, (x1, y1, w, h))
                    tracking_ball = True
                    lost_tracker_frames = 0
                    cx, cy = get_box_center(x1, y1, x2, y2)
                    ball_center = (cx, cy)
                    break
            if tracking_ball:
                break

    # Player YOLO detection every interval
    if current_time - last_player_check_time > player_recheck_interval:
        tracking_player = False
        player_tracker = None

    if tracking_player and player_tracker is not None:
        success, box = player_tracker.update(image)
        if success:
            player_lost_frames = 0
            x, y, w, h = map(int, box)
            x2, y2 = x + w, y + h
            player_tracker_box = (int(x), int(y), int(x2), int(y2))
            cv2.rectangle(image, (x, y), (x2, y2), (255, 0, 255), 2)
            cv2.putText(image, "Tracking Player", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        else:
            player_lost_frames += 1
            if player_lost_frames >= player_lost_threshold:
                tracking_player = False
                player_tracker = None
                print("🛑 Lost tracking player.")

    if not tracking_player:
        try:
            person_results = model_person(image)
            last_player_check_time = current_time
            for result in person_results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if class_id == 0 and confidence > 0.7:
                        player_boxes.append((x1, y1, x2, y2, confidence))

            if ball_center and player_boxes:
                closest = get_closest_player_to_ball(ball_center, player_boxes)
                if closest:
                    x1, y1, x2, y2 = map(int, closest[:4])
                    w, h = x2 - x1, y2 - y1
                    player_tracker = cv2.TrackerCSRT_create()
                    player_tracker.init(image, (x1, y1, w, h))
                    player_tracker_box = (x1, y1, x2, y2)
                    tracking_player = True
                    player_lost_frames = 0
        except Exception as e:
            print(f"⚠️ YOLO player detection failed: {e}")

    return tracking_ball, ball_center, [(*player_tracker_box, 0.99)] if tracking_player else []



def process_video():
    
    arm_choice = arm_to_scan
    if arm_choice not in ['left', 'right']:
        print("❌ Invalid or missing arm value. Must be 'left' or 'right'.")
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
    ball_was_near_wrist = False

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame.")
                break

            key = cv2.waitKey(1) & 0xFF
            force_redetect = (key == ord('r'))

            # Detect ball and players
            ball_detected, ball_center, player_boxes = detect_and_track_basketball_and_players(frame, force_redetect)

            # Find closest player to ball
            closest_player_box = get_closest_player_to_ball(ball_center, player_boxes)

            if closest_player_box is not None:
                x1, y1, x2, y2 = map(int, closest_player_box)
                player_crop = frame[y1:y2, x1:x2]
                player_crop_rgb = cv2.cvtColor(player_crop, cv2.COLOR_BGR2RGB)
                results = pose.process(player_crop_rgb)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    crop_w, crop_h = x2 - x1, y2 - y1

                    def map_landmark(idx):
                        lm = landmarks[idx]
                        return [lm.x * crop_w + x1, lm.y * crop_h + y1]

                    shoulder = map_landmark(SHOULDER)
                    elbow = map_landmark(ELBOW)
                    wrist = map_landmark(WRIST)

                    # Draw arm skeleton
                    cv2.line(frame, tuple(map(int, shoulder)), tuple(map(int, elbow)), (0, 255, 0), 2)
                    cv2.line(frame, tuple(map(int, elbow)), tuple(map(int, wrist)), (0, 255, 0), 2)

                    # Angle calculations
                    def calculate_angle(a, b, c):
                        a, b, c = map(np.array, (a, b, c))
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

                    smoothed_angle = savgol_filter(list(angle_buffer), window_length=7, polyorder=3)[-1] \
                        if len(angle_buffer) == angle_buffer.maxlen else shoulder_elbow_angle

                    if is_valid_shooting_pose(smoothed_angle):
                        cv2.putText(frame, "✅ Valid Pose", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        stored_release_angle, ball_was_near_wrist = store_release_angle_if_valid(
                            frame, smoothed_angle, wrist, ball_center,
                            stored_release_angle, distance_threshold=150,
                            ball_was_near_wrist=ball_was_near_wrist
                        )

                    # Display info
                    cv2.putText(frame, f"{arm_choice.title()} Elbow Angle: {int(elbow_angle)}°", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Smoothed Shoulder-Elbow Angle: {int(smoothed_angle)}°", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    if stored_release_angle is not None:
                        cv2.putText(frame, f"Stored Release Angle: {int(stored_release_angle)}°", (50, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "🕵️ Waiting for player detection...", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if not tracking_ball:
                cv2.putText(frame, "👋 Press 'R' to detect basketball", (50, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif ball_detected:
                cv2.putText(frame, "🎯 Basketball Detected!", (50, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

            cv2.putText(frame, "Press 'R' to re-detect | 'Q' to quit", (50, frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            #cv2.imshow('Release Angle Detection', frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            if key == ord('q'):
                print("🛑 Exiting...")
                break

    cap.release()
    cv2.destroyAllWindows()




if torch.cuda.is_available():
    print(f"🚀 GPU is available: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ GPU not available. Running on CPU.")

@app.route('/video')
def video_feed():
    return Response(process_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Starting pose detection stream on http://localhost:8002/video")
    app.run(host='0.0.0.0', port=8002)

