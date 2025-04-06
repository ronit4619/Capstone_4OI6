
#### imports

from collections import deque
from scipy.signal import savgol_filter
from ultralytics import YOLO
from datetime import datetime
from queue import Queue, Empty
from threading import Lock

###### Imports############
import cv2
import numpy as np
import mediapipe as mp
import time
import torch
import json
import threading
import numpy as np
import tkinter as tk
# Global Variable
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
mp_pose = mp.solutions.pose
model = YOLO('best.pt')
model.to(torch_device)
ball_queue = Queue()

# Initialize tracker state
tracker = None
tracking_ball = False
lost_tracker_frames = 0
tracker_lost_threshold = 10
last_yolo_check_time = 0
recheck_interval = 0.05  # seconds #steph curry release is 0.4

## global
last_ball_center = None
last_ball_box = None  # will store (x1, x2)

latest_release_ball_angle = None
latest_speed = None
speed_lock = Lock()

max_velocity = -1

BASE_WIDTH = 4096 


############### SMALL FUNCTIONS#########################################
def determine_facing_direction(landmarks, image_width):
    """
    Determines if the person is facing left or right based on nose and shoulders.

    Args:
        landmarks (list): List of MediaPipe pose landmarks.
        image_width (int): Width of the frame/image.

    Returns:
        str: 'left', 'right', or 'undetermined'
    """
    try:
        nose_x = landmarks[mp_pose.PoseLandmark.NOSE.value].x * image_width
        left_shoulder_x = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * image_width
        right_shoulder_x = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * image_width

        # Compare distance from nose to shoulders
        if abs(nose_x - left_shoulder_x) < abs(nose_x - right_shoulder_x):
            return 'left'  # facing left
        else:
            return 'right'  # facing right
    except Exception as e:
        print(f"Facing direction detection error: {e}")
        return 'undetermined'


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

def get_box_center(x1, y1, x2, y2):
    """
    Returns the center (cx, cy) of a bounding box.
    """
    cx = x1 + (x2 - x1) // 2
    cy = y1 + (y2 - y1) // 2
    return (cx, cy)
#################### MODEL AND Scaling #####
# def detect_and_track_basketball(image, force_redetect=False):
#     global tracker, tracking_ball, lost_tracker_frames, last_yolo_check_time

#     current_time = time.time()
#     if force_redetect:
#         tracking_ball = False
#         tracker = None
#         print("🔄 Manual reset: Forcing re-detection.")

#     # Periodic YOLO re-check every recheck_interval seconds
#     if tracking_ball and (current_time - last_yolo_check_time > recheck_interval):
#         print("🔁 Performing periodic YOLO confirmation.")
#         tracking_ball = False
#         tracker = None

#     # If currently tracking, try updating with tracker
#     if tracking_ball and tracker is not None:
#         success, box = tracker.update(image)
#         if success:
#             lost_tracker_frames = 0
#             x, y, w, h = map(int, box)
#             cx, cy = get_box_center(x, y, x + w, y + h)
#             cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(image, "Tracking", (x, y - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#             cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
#             return True, (cx, cy), (x, x + w)  # 👈 include x1, x2
#         else:
#             lost_tracker_frames += 1
#             if lost_tracker_frames >= tracker_lost_threshold:
#                 print("🛑 Tracker lost the ball. Switching to re-detection.")
#                 tracking_ball = False
#                 tracker = None
#             return False, None, None

#     # If not tracking or re-detecting
#     results = model(image)
#     last_yolo_check_time = current_time
#     for result in results:
#         for box in result.boxes:
#             class_id = int(box.cls[0])
#             confidence = float(box.conf[0])
#             if class_id == 0 and confidence > 0.70:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 w, h = x2 - x1, y2 - y1
#                 tracker = cv2.TrackerCSRT_create()
#                 tracker.init(image, (x1, y1, w, h))
#                 tracking_ball = True
#                 lost_tracker_frames = 0
#                 cx, cy = get_box_center(x1, y1, x2, y2)
#                 cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 label = f"basketball: {confidence:.2f}"
#                 cv2.putText(image, label, (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#                 cv2.circle(image, (cx, cy), 5, (0, 255, 255), -1)
#                 return True, (cx, cy), (x1, x2)  # this is key

#     return False, None, None  # when no detection

# def detect_and_track_basketball(image, force_redetect=False):
#     global tracker, tracking_ball, lost_tracker_frames, last_yolo_check_time

#     current_time = time.time()
#     last_yolo_check_time = current_time

#     best_ball_conf = 0
#     best_rim_conf = 0
#     basketball_box = None
#     basketball_cx_cy = None
#     rim_box = None

#     if force_redetect:
#         tracking_ball = False
#         tracker = None
#         print("🔄 Manual reset: Forcing re-detection.")

#     if tracking_ball and (current_time - last_yolo_check_time > recheck_interval):
#         print("🔁 Performing periodic YOLO confirmation.")
#         tracking_ball = False
#         tracker = None

#     if tracking_ball and tracker is not None:
#         success, box = tracker.update(image)
#         if success:
#             lost_tracker_frames = 0
#             x, y, w, h = map(int, box)
#             cx, cy = get_box_center(x, y, x + w, y + h)
#             cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(image, "Tracking", (x, y - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#             cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
#             return True, (cx, cy), (x, x + w), rim_box
#         else:
#             lost_tracker_frames += 1
#             if lost_tracker_frames >= tracker_lost_threshold:
#                 print("🛑 Tracker lost the ball. Switching to re-detection.")
#                 tracking_ball = False
#                 tracker = None

#     # Run YOLO detection
#     results = model(image)

#     for result in results:
#         for box in result.boxes:
#             class_id = int(box.cls[0])
#             confidence = float(box.conf[0])
#             x1, y1, x2, y2 = map(int, box.xyxy[0])

#             if class_id == 0 and confidence > best_ball_conf:
#                 best_ball_conf = confidence
#                 w, h = x2 - x1, y2 - y1
#                 tracker = cv2.TrackerCSRT_create()
#                 tracker.init(image, (x1, y1, w, h))
#                 tracking_ball = True
#                 lost_tracker_frames = 0
#                 cx, cy = get_box_center(x1, y1, x2, y2)
#                 basketball_box = (x1, x2)
#                 basketball_cx_cy = (cx, cy)

#                 cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 label = f"basketball: {confidence:.2f}"
#                 cv2.putText(image, label, (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#                 cv2.circle(image, (cx, cy), 5, (0, 255, 255), -1)

#             elif class_id == 1 and confidence > best_rim_conf:
#                 best_rim_conf = confidence
#                 rim_box = (x1, y1, x2, y2)
#                 cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)
#                 cv2.putText(image, f"rim: {confidence:.2f}", (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

#     if basketball_cx_cy:
#         return True, basketball_cx_cy, basketball_box, rim_box
#     else:
#         return False, None, None, rim_box


def detect_and_track_basketball(image, force_redetect=False):
    global last_yolo_check_time

    current_time = time.time()
   

    run_yolo = force_redetect or (current_time - last_yolo_check_time > recheck_interval)
    
    if not run_yolo:
        return False, None, None, None

    last_yolo_check_time = current_time

    best_ball_conf = 0
    best_rim_conf = 0
    basketball_box = None
    basketball_cx_cy = None
    rim_box = None

    results = model(image)

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if class_id == 0 and confidence > best_ball_conf:
                best_ball_conf = confidence
                cx, cy = get_box_center(x1, y1, x2, y2)
                basketball_box = (x1, x2)
                basketball_cx_cy = (cx, cy)

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"basketball: {confidence:.2f}"
                cv2.putText(image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(image, (cx, cy), 5, (0, 255, 255), -1)

            elif class_id == 1 and confidence > best_rim_conf:
                best_rim_conf = confidence
                rim_box = (x1, y1, x2, y2)
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(image, f"rim: {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    if basketball_cx_cy:
        return True, basketball_cx_cy, basketball_box, rim_box
    else:
        return False, None, None, rim_box



def determine_dynamic_scale(x1, x2,real_diameter_m=0.25):
    """
    Calculates pixels-per-meter scale using basketball diameter.

    Args:
        x1 (int): Left x-coordinate of bounding box.
        x2 (int): Right x-coordinate of bounding box.
        real_diameter_m (float): Real-world diameter of basketball (default: 0.24m).

    Returns:
        float: Scale in pixels per meter.
    """
    pixel_diameter = abs(x2 - x1) 
    
    if pixel_diameter == 0:
        return None  # Avoid division by zero
    return pixel_diameter / (real_diameter_m )

###############Shooting Mechanics##########################

def store_release_angle_if_valid(frame, smoothed_angle, wrist, ball_center,
                                  stored_release_angle, distance_threshold=150,
                                  ball_was_near_wrist=False, Newscale=100,speed=None):
    """
    Detects release and returns updated release angle, state, and projectile points (if any).
    """
    if None in [wrist, ball_center]:
        return stored_release_angle, ball_was_near_wrist, []

    distance = calculate_distance(ball_center, wrist) / Newscale

    if (distance < distance_threshold ):
        ball_was_near_wrist = True

    elif ball_was_near_wrist:
        stored_release_angle = smoothed_angle
        ball_was_near_wrist = False

        if stored_release_angle is not None:
            #log_release_angle_to_json(stored_release_angle)
            cv2.putText(frame, f"Release Angle: {int(stored_release_angle)}°",
                        (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Generate and return the arc points to store
            # if wrist is not None:
            #     start_pos = tuple(map(int, wrist))
            #     new_projectile = generate_projectile_points(stored_release_angle,v=speed, start_pos=start_pos, scale=Newscale)

    # Show distance always
    cv2.putText(frame, f"Ball-Wrist Dist: {int(distance)} px",
                (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    return stored_release_angle, ball_was_near_wrist     #, new_projectile
 
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


#################### Projectile##############################

def generate_projectile_points(release_angle_deg, v=7.9, g=9.81, scale=135, start_pos=(100, 400), rim=None):
    import numpy as np
    import json

    angle_rad = np.radians(release_angle_deg)
    t_vals = np.linspace(0, 2 * v * np.sin(angle_rad) / g, num=100)
    points = []
    ball_made = False
    hit_above = False
    hit_inside = False

    # Parse rim if provided
    if rim:
        rim_left, rim_top, rim_right, rim_bottom = rim

    for t in t_vals:
        x = v * np.cos(angle_rad) * t
        y = v * np.sin(angle_rad) * t - 0.5 * g * t**2
        x_px = int(start_pos[0] + x * scale)
        y_px = int(start_pos[1] - y * scale)
        points.append((x_px, y_px))

        # Check if this point "goes in" the hoop
        if rim:
            # 1️⃣ Point passed above rim (entry path)
            if rim_left <= x_px <= rim_right and y_px <= rim_top:
                hit_above = True

            # 2️⃣ Point inside rim box
            if rim_left <= x_px <= rim_right and rim_top <= y_px <= rim_bottom:
                hit_inside = True

        if rim and hit_above and hit_inside:
            ball_made = True
            

    # Logging
    # log_data = {
    #     "launch_angle_deg": round(release_angle_deg, 2),
    #     "velocity_mps": round(v, 2),
    #     "scale_px_per_m": round(scale, 2),
    #     "ball_made": ball_made
    # }

    # filename = "generate_projectile_points_log.json"
    # with open(filename, "w") as f:
    #     json.dump(log_data, f, indent=2)

    return points, ball_made


def average_initial_velocity(ball_positions):
    """
    Calculates the initial velocity (in m/s) using both x and y directions
    from the last two tracked ball positions.
    """
    if ball_positions is None or len(ball_positions) < 2:
        return None

    (t1, (x1, y1), scale1), (t2, (x2, y2), scale2) = list(ball_positions)[-2:]

    dt = t2 - t1
    if dt <= 0:
        return None

    dx_px = x2 - x1
    dy_px = y2 - y1
    avg_scale = (scale1 + scale2) / 2

    dx_m = dx_px / avg_scale
    dy_m = dy_px / avg_scale

    vx = dx_m / dt
    vy = dy_m / dt

    ball_angle =  np.degrees (np.arctan2(vy, vx))
    v = (vx ** 2 + vy ** 2) ** 0.5

    # Log everything
    # log_release_angle_to_json(
    #     release_angle=None,
    #     speed=v,
    #     ball_positions=[(t1, (x1, y1)), (t2, (x2, y2))],
    #     px_per_m=avg_scale
    # )

    if ball_angle < 0:
        ball_angle += 360

# Then limit to the 0°–90° range if you only care about upward release angles
    if ball_angle > 90:
        ball_angle = 180 - ball_angle

    return v, ball_angle


def velocity_consumer():
    global latest_speed
    global latest_release_ball_angle
    ball_position_buffer = deque(maxlen=60)

    while True:
        try:
            item = ball_queue.get(timeout=1)
            if item == "STOP":
                break

            video_time, ball_center,dynamic_scale = item
            ball_position_buffer.append((video_time, ball_center,dynamic_scale))

            if len(ball_position_buffer) >= 2:
                avg_vel,ball_angle = average_initial_velocity(ball_position_buffer)
                if avg_vel is not None:
                    with speed_lock:
                        latest_speed = avg_vel
                        latest_release_ball_angle = ball_angle  
        except Empty:
            continue


################################LOGGER###################################
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

def log_release_angle_to_json(release_angle=None, speed=None, ball_positions=None, px_per_m=None, filename="release_spped_time.json"):
    """
    Logs release angle, speed, ball positions, and px_per_m into a JSON file.
    """
    data = []

    try:
        with open(filename, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    entry = {
        "timestamp": datetime.now().isoformat()
    }

    if release_angle is not None:
        entry["release_angle"] = round(release_angle, 2)

    if speed is not None:
        entry["release_speed_mps"] = round(speed, 3)

    if ball_positions is not None:
        entry["ball_positions"] = [
            {
                "time": t,
                "x": int(xy[0]),
                "y": int(xy[1])
            } for t, xy in ball_positions
        ]

    if px_per_m is not None:
        entry["px_per_m"] = round(px_per_m, 2)

    data.append(entry)

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

def log_release_speed_to_json(speed, filename="release_speeds.json"):
    data = []

    # Try loading existing data
    try:
        with open(filename, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass  # File doesn't exist or is malformed

    # Append new entry
    data.append({
        "timestamp": datetime.now().isoformat(),
        "average_release_speed_mps": round(speed, 3)
    })

    # Save to file
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
##############################################################################
def get_screen_resolution():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    return screen_width, screen_height

def draw_scaled_text(frame, text, pos, font_scale_base, color, thickness_base, scale_factor):
    font_scale = font_scale_base * scale_factor
    thickness = max(1, int(thickness_base * scale_factor))
    pos = (int(pos[0] * scale_factor), int(pos[1] * scale_factor))
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def process_video():
    start_time = time.time()
    max_velocity = -1
    projectile_points = []
    
    arm_choice = input("Which arm would you like to track? Type 'left' or 'right': ").strip().lower()
    if arm_choice not in ['left', 'right']:
        print("Invalid choice. Please enter 'left' or 'right'.")
        return

    SHOULDER = getattr(mp_pose.PoseLandmark, f"{arm_choice.upper()}_SHOULDER").value
    ELBOW = getattr(mp_pose.PoseLandmark, f"{arm_choice.upper()}_ELBOW").value
    WRIST = getattr(mp_pose.PoseLandmark, f"{arm_choice.upper()}_WRIST").value
    non_dominant = 'right' if arm_choice == 'left' else 'left'
    NON_DOM_SHOULDER = getattr(mp_pose.PoseLandmark, f"{non_dominant.upper()}_SHOULDER").value

    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    print("Webcam opened successfully!")
    print("Press 'R' to re-detect basketball | 'Q' to quit")
    print(fps)
    stored_release_angle = None
    angle_buffer = deque(maxlen=7)

    screen_w, screen_h = get_screen_resolution()
    target_w = int(screen_w * 0.9)
    target_h = int(screen_h * 0.9)
    BASE_WIDTH = 3840

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        ball_was_near_wrist = False
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break
            frame = cv2.resize(frame, (target_w, target_h))
            scale_factor = frame.shape[1] / BASE_WIDTH
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            key = cv2.waitKey(1) & 0xFF
            force_redetect = (key == ord('r'))

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                shoulder = [landmarks[SHOULDER].x * frame.shape[1], landmarks[SHOULDER].y * frame.shape[0]]
                elbow = [landmarks[ELBOW].x * frame.shape[1], landmarks[ELBOW].y * frame.shape[0]]
                wrist = [landmarks[WRIST].x * frame.shape[1], landmarks[WRIST].y * frame.shape[0]]
                dominant_shoulder_z = landmarks[SHOULDER].z
                non_dominant_shoulder_z = landmarks[NON_DOM_SHOULDER].z

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

                if dominant_shoulder_z > non_dominant_shoulder_z:
                    smoothed_angle = 180 - smoothed_angle

                ball_detected, ball_center, ball_box_x, rim_box = detect_and_track_basketball(frame, force_redetect)

                if ball_detected and ball_center and ball_box_x:
                    last_ball_box = ball_box_x
                    x1, x2 = last_ball_box
                    dynamic_scale = determine_dynamic_scale(x1, x2)

                    if (
                        ball_center[1] < shoulder[1] and
                        calculate_distance(ball_center, shoulder) / dynamic_scale >= 0.3
                    ):
                        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                        video_time = time.time() - start_time
                        ball_queue.put((video_time, ball_center, dynamic_scale))
                        x, y = int(ball_center[0]), int(ball_center[1])
                        size = 8
                        color = (0, 255, 0)
                        cv2.line(frame, (x - size, y - size), (x + size, y + size), color, 2)
                        cv2.line(frame, (x - size, y + size), (x + size, y - size), color, 2)

                    draw_scaled_text(frame, "Basketball Detected!", (50, 200), 0.6, (0, 165, 255), 1.5, scale_factor)
                    draw_scaled_text(frame, f"Scale: {dynamic_scale:.2f} px/m", (50, 230), 0.6, (255, 255, 0), 1.5, scale_factor)

                    if is_valid_shooting_pose:
                        draw_scaled_text(frame, "valid pose", (50, 350), 0.6, (0, 255, 0), 1.5, scale_factor)

                        with speed_lock:
                            current_speed = latest_speed

                        stored_release_angle, ball_was_near_wrist = store_release_angle_if_valid(
                            frame, smoothed_angle, wrist, ball_center,
                            stored_release_angle, distance_threshold=0.3,
                            ball_was_near_wrist=ball_was_near_wrist, Newscale=dynamic_scale, speed=current_speed)

                        if stored_release_angle is not None:
                            start_pos = tuple(map(int, wrist))
                            with speed_lock:
                                current_speed = latest_speed
                                current_angle = latest_release_ball_angle

                            if current_speed is not None:
                                if (calculate_distance(wrist, ball_center) / dynamic_scale) < 0.1 and ball_center[1] > shoulder[1]:
                                    max_velocity = -1

                                if max_velocity <= current_speed:
                                    max_velocity = current_speed
                                    projectile_points, make = generate_projectile_points(
                                        current_angle,
                                        v=current_speed,
                                        start_pos=start_pos,
                                        scale=dynamic_scale,
                                        rim=rim_box
                                    )
                                draw_scaled_text(frame, f"Max Speed: {max_velocity:.2f} m/s", (50, 400), 0.6, (0, 255, 255), 1.5, scale_factor)

                                if make:
                                    overlay = frame.copy()
                                    flash_color = (0, 255, 0)
                                    alpha = 0.4
                                    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), flash_color, -1)
                                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                                    draw_scaled_text(frame, "Shot Made!", (50, 50), 1.2, (255, 255, 255), 3, scale_factor)

                    for (x, y) in projectile_points:
                        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

                elif not tracking_ball:
                    draw_scaled_text(frame, "Press 'R' to detect basketball", (50, 200), 1.0, (0, 0, 255), 2, scale_factor)

                draw_scaled_text(frame, f"{arm_choice.title()} Elbow Angle: {int(elbow_angle)}°", (50, 50), 1.0, (0, 255, 0), 2, scale_factor)
                draw_scaled_text(frame, f"Smoothed Shoulder-Elbow Angle: {int(smoothed_angle)}°", (50, 100), 1.0, (255, 255, 0), 2, scale_factor)

                if stored_release_angle is not None:
                    draw_scaled_text(frame, f"Stored Release Angle: {int(stored_release_angle)}°", (50, 150), 1.0, (0, 0, 255), 2, scale_factor)

            else:
                draw_scaled_text(frame, "Waiting for pose detection...", (50, 50), 0.8, (0, 0, 255), 2, scale_factor)

            draw_scaled_text(frame, "Press 'R' to re-detect | 'Q' to quit", (50, frame.shape[0] - 30), 0.6, (255, 255, 255), 1.2, scale_factor)

            
            cv2.imshow('Release Angle Detection', frame)

            if key == ord('q'):
                print("Exiting...")
                break

    ball_queue.put("STOP")
    consumer_thread.join()
    cap.release()
    cv2.destroyAllWindows()









if torch.cuda.is_available():
    print(f"GPU is available: {torch.cuda.get_device_name(0)}")
else:
    print("GPU not available. Running on CPU.")


# Start the consumer thread BEFORE processing video
consumer_thread = threading.Thread(target=velocity_consumer)
consumer_thread.start()
process_video()