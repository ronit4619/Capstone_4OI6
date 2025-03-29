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


def generate_projectile_points(release_angle_deg, v=5.5, g=9.81, scale=135, start_pos=(100, 400)):
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
                                  ball_was_near_wrist=False, Newscale=100):
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



            release_time = time.time()
            release_position = ball_center


            later_time = time.time()
            later_position = ball_center  #current ball position

            speed = calculate_release_speed(release_position, release_time, later_position, later_time, Newscale)

            # ✅ Generate and return the arc points to store
            if wrist is not None:
                start_pos = tuple(map(int, wrist))
                new_projectile = generate_projectile_points(stored_release_angle, start_pos=start_pos, scale=Newscale)

    # Show distance always
    cv2.putText(frame, f"Ball-Wrist Dist: {int(distance)} px",
                (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    return stored_release_angle, ball_was_near_wrist, new_projectile




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
#             return True, (cx, cy)
#         else:
#             lost_tracker_frames += 1
#             if lost_tracker_frames >= tracker_lost_threshold:
#                 print("🛑 Tracker lost the ball. Switching to re-detection.")
#                 tracking_ball = False
#                 tracker = None
#             return False, None

#     # If not tracking or re-detecting
#     results = model(image)
#     last_yolo_check_time = current_time  # reset timer on detection
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
#                 return True, (cx, cy)

#     return False, None ## the return type is boolean for detection and the centre of ther ball



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





def calculate_release_speed(pos1, t1, pos2, t2, pixels_per_meter):
    """
    Calculates horizontal release speed (m/s) between two points.

    Args:
        pos1 (tuple): (x, y) of ball at release.
        t1 (float): Timestamp of release (seconds).
        pos2 (tuple): (x, y) of ball shortly after release.
        t2 (float): Timestamp of second point (seconds).
        pixels_per_meter (float): Pixel-to-meter scale based on ball diameter.

    Returns:
        float: Estimated horizontal speed in m/s.
    """
    if None in [pos1, pos2, t1, t2] or t2 <= t1:
        return None

    dx_px = pos2[0] - pos1[0]
    dx_m = dx_px / pixels_per_meter
    dt = t2 - t1

    return dx_m / dt





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

    cap = cv2.VideoCapture("C:/Users/antho/Downloads/IMG_0523.MOV")
    if not cap.isOpened():
        print("❌ Error: Could not access webcam.")
        return

    print("✅ Webcam opened successfully!")
    print("🎮 Press 'R' to re-detect basketball | 'Q' to quit")
    stored_release_angle = None
    angle_buffer = deque(maxlen=7)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
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
    ball_was_near_wrist=ball_was_near_wrist,Newscale=dynamic_scale)
                       
                       


                       

                       
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