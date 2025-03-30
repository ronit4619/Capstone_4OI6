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

import random


torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'


mp_pose = mp.solutions.pose
model = YOLO('best.pt')
model.to(torch_device)

# Initialize tracker state
tracker = None
tracking_ball = False
lost_tracker_frames = 0
tracker_lost_threshold = 10
last_yolo_check_time = 0
recheck_interval = 0.1  # seconds #steph curry release is 0.4

## global
last_ball_center = None
last_ball_box = None  # will store (x1, x2)


import numpy as np

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



def generate_projectile_points(release_angle_deg, v=7.9, g=9.81, scale=135, start_pos=(100, 400)):
    angle_rad = np.radians(release_angle_deg)
    t_vals = np.linspace(0, 2 * v * np.sin(angle_rad) / g, num=100)
    points = []

    for t in t_vals:
        x = v * np.cos(angle_rad) * t
        y = v * np.sin(angle_rad) * t - 0.5 * g * t**2
        x_px = int(start_pos[0] + x * scale)
        y_px = int(start_pos[1] - y * scale)

        points.append((x_px, y_px))

    return points









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


def quick_detect_ball(image, max_attempts=10, delay=0.01, confidence_threshold=0.7):
    """
    Detects the basketball using YOLO and waits until it is found or max_attempts is reached.

    Args:
        image (ndarray): The current video frame.
        max_attempts (int): Max retries before giving up.
        delay (float): Delay in seconds between attempts.
        confidence_threshold (float): Minimum confidence to accept detection.

    Returns:
        tuple or None: (x, y) center of basketball if found, else None.
    """
    for _ in range(max_attempts):
        results = model(image)

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                if class_id == 0 and confidence > confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = x1 + (x2 - x1) // 2
                    cy = y1 + (y2 - y1) // 2
                    return (cx, cy)

        time.sleep(delay)

    return None  # Ball wasn't detected within allowed attempts

def store_release_angle_if_valid(frame, smoothed_angle, wrist, ball_center,
                                  stored_release_angle, distance_threshold=150,
                                  ball_was_near_wrist=False, Newscale=100,ball_positions=None):
    """
    Detects release and returns updated release angle, state, and projectile points (if any).
    """
    if None in [wrist, ball_center]:
        return stored_release_angle, ball_was_near_wrist, []

    distance = calculate_distance(ball_center, wrist)
    new_projectile = []

    if distance < 100:
        ball_was_near_wrist = True

    elif ball_was_near_wrist and distance >= distance_threshold:
        stored_release_angle = smoothed_angle
        ball_was_near_wrist = False

        if stored_release_angle is not None:
            log_release_angle_to_json(stored_release_angle)
            cv2.putText(frame, f"Release Angle: {int(stored_release_angle)}°",
                        (50, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)




            speed=average_x_velocity(ball_positions, px_per_m=Newscale)
            # ✅ Generate and return the arc points to store
            if wrist is not None:
                start_pos = tuple(map(int, wrist))
                new_projectile = generate_projectile_points(stored_release_angle,v=speed, start_pos=start_pos, scale=Newscale)

    # Show distance always
    cv2.putText(frame, f"Ball-Wrist Dist: {int(distance)} px",
                (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    return stored_release_angle, ball_was_near_wrist, new_projectile



def average_x_velocity(ball_positions, px_per_m=100, max_pairs=4):
    """
    Calculates average horizontal velocity (x-direction) in m/s,
    and logs it with ball positions and scale.
    """
    if ball_positions is None or len(ball_positions) < 2:
        return None

    recent = list(ball_positions)[-max_pairs:]

    total_dx = 0
    total_dt = 0

    for i in range(1, len(recent)):
        t1, (x1, _) = recent[i - 1]
        t2, (x2, _) = recent[i]
        dt = t2 - t1
        if dt > 0:
            dx_px = x2 - x1
            total_dx += dx_px
            total_dt += dt

    if total_dt == 0:
        return None

    dx_m = total_dx / px_per_m
    avg_velocity = abs(dx_m / total_dt)

    # Log everything including px_per_m
    log_release_angle_to_json(
        release_angle=None,
        speed=avg_velocity,
        ball_positions=recent,
        px_per_m=px_per_m
    )

    return avg_velocity


# def smart_average_velocity(ball_positions, shoulder, px_per_m=100, min_dist_m=0.3, samples=4):
#     """
#     Calculates average horizontal velocity in m/s by sampling points
#     where the ball is at least `min_dist_m` meters away from the shoulder.

#     Logs the speed, positions used, and px_per_m.
#     """
#     valid = [
#         (t, (x, y)) for t, (x, y) in ball_positions
#         if calculate_distance((x, y), shoulder) >= min_dist_m * px_per_m
#     ]

#     # ✅ Ensure at least 2 valid points
#     if len(valid) < 2:
#         return None

#     # Sample safely: at least 2, at most all
#     n_to_sample = min(samples, len(valid))
#     if n_to_sample < 2:
#         return None

#     sampled = random.sample(valid, n_to_sample)
#     sampled.sort(key=lambda x: x[0])  # sort by time

#     total_dx = 0
#     total_dt = 0
#     for i in range(1, len(sampled)):
#         t1, (x1, _) = sampled[i - 1]
#         t2, (x2, _) = sampled[i]
#         dt = t2 - t1
#         if dt > 0:
#             total_dx += x2 - x1
#             total_dt += dt

#     if total_dt == 0:
#         return None

#     dx_m = total_dx / px_per_m
#     avg_velocity = abs(dx_m / total_dt)

#     log_release_angle_to_json(
#         release_angle=None,
#         speed=avg_velocity,
#         ball_positions=sampled,
#         px_per_m=px_per_m
#     )

#     return avg_velocity



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

    # If currently tracking, try updating with tracker
    if tracking_ball and tracker is not None:
        success, box = tracker.update(image)
        if success:
            lost_tracker_frames = 0
            x, y, w, h = map(int, box)
            cx, cy = get_box_center(x, y, x + w, y + h)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, "Tracking", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
            return True, (cx, cy), (x, x + w)  # 👈 include x1, x2
        else:
            lost_tracker_frames += 1
            if lost_tracker_frames >= tracker_lost_threshold:
                print("🛑 Tracker lost the ball. Switching to re-detection.")
                tracking_ball = False
                tracker = None
            return False, None, None

    # If not tracking or re-detecting
    results = model(image)
    last_yolo_check_time = current_time
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            if class_id == 0 and confidence > 0.70:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                tracker = cv2.TrackerCSRT_create()
                tracker.init(image, (x1, y1, w, h))
                tracking_ball = True
                lost_tracker_frames = 0
                cx, cy = get_box_center(x1, y1, x2, y2)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"basketball: {confidence:.2f}"
                cv2.putText(image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(image, (cx, cy), 5, (0, 255, 255), -1)
                return True, (cx, cy), (x1, x2)  # 👈 this is key

    return False, None, None  # when no detection


def determine_dynamic_scale(x1, x2, real_diameter_m=0.24):
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
    return pixel_diameter / real_diameter_m




def process_video():

    projectile_points = []
    arm_choice = input("👉 Which arm would you like to track? Type 'left' or 'right': ").strip().lower()
    if arm_choice not in ['left', 'right']:
        print("❌ Invalid choice. Please enter 'left' or 'right'.")
        return

    SHOULDER = getattr(mp_pose.PoseLandmark, f"{arm_choice.upper()}_SHOULDER").value
    ELBOW = getattr(mp_pose.PoseLandmark, f"{arm_choice.upper()}_ELBOW").value
    WRIST = getattr(mp_pose.PoseLandmark, f"{arm_choice.upper()}_WRIST").value

    non_dominant = 'right' if arm_choice == 'left' else 'left'
    NON_DOM_SHOULDER = getattr(mp_pose.PoseLandmark, f"{non_dominant.upper()}_SHOULDER").value
    
    #cap = cv2.VideoCapture("")
    #"C:/Users/antho/Downloads/20250325_102838.mp4"
    #"C:/Users/antho/Downloads/IMG_0519.MOV"

    cap = cv2.VideoCapture("C:/Users/antho/Downloads/IMG_0526.MOV")
    if not cap.isOpened():
        print("❌ Error: Could not access webcam.")
        return

    print("✅ Webcam opened successfully!")
    print("🎮 Press 'R' to re-detect basketball | 'Q' to quit")
    stored_release_angle = None
    angle_buffer = deque(maxlen=7)
    ball_position_buffer = deque(maxlen=20)  # stores (time, (x, y)) tuples




    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        # start global variable
        #shooting_state = 'ready'
        ball_was_near_wrist = False
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

                ## added
                direction = determine_facing_direction(landmarks, frame.shape[1]) 
                cv2.putText(frame, f"Facing: {direction}", (50, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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
                
                # Compare z-coordinates
                if dominant_shoulder_z > non_dominant_shoulder_z:
                    smoothed_angle=180-smoothed_angle
                
                
                #if direction == 'right':
                #    smoothed_angle = 180 - smoothed_angle

               
                ball_detected, ball_center,ball_box_x = detect_and_track_basketball(frame, force_redetect)

                if ball_detected and ball_center and ball_box_x:
                    last_ball_center = ball_center
                    last_ball_box = ball_box_x
                    x1, x2 = last_ball_box
                    dynamic_scale = determine_dynamic_scale(x1, x2)


                    # recently added
                    if (
                        ball_center[1] < shoulder[1] and  # Ball is above shoulder
                        calculate_distance(ball_center, shoulder) * dynamic_scale >= 1 * dynamic_scale  # Ball is 0.3m away
                    ):
                        ball_position_buffer.append((time.time(), ball_center))


                    cv2.putText(frame, "🎯 Basketball Detected!", (50, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                     # ✅ Display scale on screen
                    cv2.putText(frame, f"Scale: {dynamic_scale:.2f} px/m", (50, 230),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                   

                    if is_valid_shooting_pose :
                       # just call the function
                       position = (50, 180)
                       cv2.putText(frame, "valid pose", position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                       stored_release_angle, ball_was_near_wrist, new_points = store_release_angle_if_valid(
    frame, smoothed_angle, wrist, ball_center,
    stored_release_angle, distance_threshold=150,
    ball_was_near_wrist=ball_was_near_wrist,Newscale=dynamic_scale,ball_positions=ball_position_buffer)
                       
                       


                       

                       
                    if new_points:
                        projectile_points = new_points   
                    for (x, y) in projectile_points:
                        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)




                       
                       
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



if torch.cuda.is_available():
    print(f"🚀 GPU is available: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ GPU not available. Running on CPU.")

process_video()