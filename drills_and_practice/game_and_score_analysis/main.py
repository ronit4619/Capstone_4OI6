import cv2
import numpy as np
from ultralytics import YOLO
import torch
import time
from collections import defaultdict, deque
import os
from tqdm import tqdm
import argparse
import threading
from queue import Queue
from math import acos, degrees, sqrt


#shot from left, right, center three point.

#see accuracy from each spot

#see accuracy for free throw
#see accuracy for three pointer

#python .\main.py --mode video --input Shot_1.mp4 --output output2.avi
#python .\main.py --mode realtime --output output2.avi

#opencv-python
#numpy
#ffmpeg-python
#torch
#torchvision
#torchaudio
#torchmetrics
#ultralytics
#scikit-learn
#mediapipe
#pandas
#matplotlib
#seaborn


# Define keypoint indices for pose estimation
HEAD = 0
RIGHT_SHOULDER = 6; RIGHT_ELBOW = 8; RIGHT_WRIST = 10
RIGHT_HIP = 12; RIGHT_KNEE = 14; RIGHT_ANKLE = 16
LEFT_SHOULDER = 5; LEFT_ELBOW = 7; LEFT_WRIST = 9
LEFT_HIP = 11; LEFT_KNEE = 13; LEFT_ANKLE = 15

def joint_angle(A, B, C):
    """Calculate the angle (in degrees) at joint B formed by segments BA and BC."""
    if any(p is None for p in [A, B, C]):
        return 0.0
    BAx, BAy = A[0] - B[0], A[1] - B[1]  # vector from B to A
    BCx, BCy = C[0] - B[0], C[1] - B[1]  # vector from B to C
    dot = BAx * BCx + BAy * BCy         # dot product of BA and BC
    magBA = sqrt(BAx**2 + BAy**2)
    magBC = sqrt(BCx**2 + BCy**2)
    if magBA == 0 or magBC == 0:
        return 0.0  # avoid division by zero
    cos_angle = max(-1.0, min(1.0, dot / (magBA * magBC)))
    angle_rad = acos(cos_angle)
    return degrees(angle_rad)

class BallTracker:
    def __init__(self):
        self.positions = []
        self.current_position = None
        self.last_position = None
        self.last_detection_time = 0
        self.max_positions = 30  # Reduced from 30 to 15 positions
        
    def update(self, detections):
        if not detections:
            return None
            
        # Get the detection with highest confidence
        best_detection = max(detections, key=lambda x: x['conf'])
        
        # Update positions
        x1, y1, x2, y2 = best_detection['bbox']
        center = (int((x1 + x2) // 2), int((y1 + y2) // 2))
        
        # Store position
        self.last_position = self.current_position
        self.current_position = center
        
        if len(self.positions) > self.max_positions:
            self.positions.pop(0)
        self.positions.append(center)
        
        return best_detection
        
    def get_trajectory_info(self):
        """Return trajectory information for shot detection."""
        if len(self.positions) < 3:
            return None
            
        return {
            'points': self.positions.copy(),
            'positions': self.positions.copy()  # For compatibility with existing code
        }
        
    def clear_trajectory(self):
        """Clear the trajectory history"""
        self.positions = []
        self.current_position = None
        self.last_position = None

class HoopTracker:
    def __init__(self, fps):
        self.fps = fps  # Store actual video FPS
        self.hoop_found = False
        self.hoop_position = None  # (x, y, w, h)
        self.hoop_center = None    # (x, y)
        
        # Box states
        self.entry_box_touched = False
        self.make_box_touched = False
        self.entry_box_touch_frame = None
        self.entry_box_exit_frame = None
        
        # Frame counting
        self.current_frame = 0
        
        # Time-based thresholds in seconds
        self.max_time_between_boxes = 1.0  # Maximum time between entry and make box touches
        
        # Convert to frames based on FPS
        self.max_frames_between_boxes = int(self.max_time_between_boxes * self.fps)
        
    def reset_box_states(self):
        """Reset box states and timing."""
        self.entry_box_touched = False
        self.make_box_touched = False

    def bboxes_overlap(self, bbox1, bbox2):
        """Check if two bounding boxes overlap."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        overlap_x = (x1_min <= x2_max) and (x2_min <= x1_max)
        overlap_y = (y1_min <= y2_max) and (y2_min <= y1_max)
        return overlap_x and overlap_y

    def draw_detection_boxes(self, frame):
        """Draw detection boxes with colors based on their current state."""
        if not self.hoop_found:
            return frame
            
        # Get hoop dimensions
        hoop_width = self.hoop_position[2]
        hoop_height = self.hoop_position[3]
        hoop_center_x = self.hoop_center[0]
        hoop_center_y = self.hoop_center[1]
        
        # Draw hoop boundary box in yellow
        hoop_box = (
            int(hoop_center_x - hoop_width/2),
            int(hoop_center_y - hoop_height/2),
            int(hoop_center_x + hoop_width/2),
            int(hoop_center_y + hoop_height/2)
        )
        cv2.rectangle(frame,
                     (hoop_box[0], hoop_box[1]),
                     (hoop_box[2], hoop_box[3]),
                     (0, 255, 255), 2)  # Yellow color in BGR
        
        # Create entry detection box
        entry_box_width = hoop_width * 0.5
        entry_box = (
            int(hoop_center_x - entry_box_width/2),
            int(hoop_center_y - entry_box_width * 2),
            int(hoop_center_x + entry_box_width/2),
            int(hoop_center_y - entry_box_width)
        )
        
        # Create make detection box
        make_box_width = hoop_width
        make_box_height = hoop_height * 0.5
        make_box = (
            int(hoop_center_x - make_box_width/2),
            int(hoop_center_y),
            int(hoop_center_x + make_box_width/2),
            int(hoop_center_y + make_box_height)
        )
        
        # Draw entry box (green if touched, white otherwise)
        entry_color = (0, 255, 0) if self.entry_box_touched else (255, 255, 255)
        cv2.rectangle(frame, 
                     (entry_box[0], entry_box[1]), 
                     (entry_box[2], entry_box[3]), 
                     entry_color, 2)
        
        # Draw make box (green if touched, white otherwise)
        make_color = (0, 255, 0) if self.make_box_touched else (255, 255, 255)
        cv2.rectangle(frame, 
                     (make_box[0], make_box[1]), 
                     (make_box[2], make_box[3]), 
                     make_color, 2)
        
        return frame
    
    def detect_shot_outcome(self, trajectory_info, is_timeout=False):
        """Detect if a shot was made using rule-based approach with two key detection boxes."""
        if not trajectory_info or not self.hoop_found:
            return None
            
        # Get trajectory points and ball bbox
        points = trajectory_info['points']
        ball_bbox = trajectory_info.get('current_bbox')
        if len(points) < 2 or not ball_bbox:
            return None
            
        # Get hoop dimensions
        hoop_width = self.hoop_position[2]
        hoop_height = self.hoop_position[3]
        hoop_center_x = self.hoop_center[0]
        hoop_center_y = self.hoop_center[1]
        
        # Create entry detection box
        entry_box_width = hoop_width * 0.5
        entry_box = (
            int(hoop_center_x - entry_box_width/2),
            int(hoop_center_y - entry_box_width),
            int(hoop_center_x + entry_box_width/2),
            int(hoop_center_y)
        )

        # Create make detection box
        make_box_width = hoop_width
        make_box_height = hoop_height * 0.5
        make_box = (
            int(hoop_center_x - make_box_width/2),
            int(hoop_center_y),
            int(hoop_center_x + make_box_width/2),
            int(hoop_center_y + make_box_height)
        )

        # Create full hoop box (area around the hoop to track when ball completely leaves)
        hoop_box = (
            int(hoop_center_x - hoop_width),  # Wider area to track
            int(hoop_center_y - hoop_height),
            int(hoop_center_x + hoop_width),
            int(hoop_center_y + hoop_height)
        )
        
        # Get current ball position
        curr_x, curr_y = points[-1]  # Use most recent point
        
        # If make box is touched before entry box and not in timeout, reset states
        if not self.entry_box_touched and self.bboxes_overlap(ball_bbox, make_box) and not is_timeout:
            self.reset_box_states()
            return None
        
        # Check for entry box contact
        if not self.entry_box_touched and self.bboxes_overlap(ball_bbox, entry_box):
            print("Ball touched entry box")  # Debug print
            self.entry_box_touched = True
        
        # If in timeout and entry box was touched, check if make box was ever touched
        if is_timeout and self.entry_box_touched:
            if self.bboxes_overlap(ball_bbox, make_box):
                print("Ball touched make box during timeout check - MADE")  # Debug print
                return 'made'
            elif not self.bboxes_overlap(ball_bbox, hoop_box):
                print("Ball left hoop area during timeout check - MISS")  # Debug print
                return 'missed'
            return None
        
        # Check if ball has completely left hoop area after touching entry box
        if self.entry_box_touched and not self.make_box_touched and not self.bboxes_overlap(ball_bbox, hoop_box):
            # Ball has left hoop area without touching make box - it's a miss
            print("Ball left hoop area - MISS")  # Debug print
            self.reset_box_states()
            return 'missed'
        
        # Check for make box contact only if entry box was touched first
        if self.entry_box_touched and not self.make_box_touched and self.bboxes_overlap(ball_bbox, make_box):
            print("Ball touched make box - MADE")  # Debug print
            self.make_box_touched = True
            # Immediately assess as a made shot
            self.reset_box_states()
            return 'made'
        
        # Increment frame counter
        self.current_frame += 1
        return None

    def update_hoop(self, detections):
        if not self.hoop_found and detections:
            # Find the rim with highest confidence
            best_rim = max(detections, key=lambda x: x['conf'])
            x1, y1, x2, y2 = best_rim['bbox']
            self.hoop_position = (x1, y1, x2-x1, y2-y1)
            self.hoop_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            self.hoop_found = True
    
class BasketballTracker:
    def __init__(self, video_path=None, output_path="/output"):
        # Initialize video capture and output
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
        else:
            self.cap = None
        self.output_path = output_path
        
        # Get video properties
        if self.cap:
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            if self.video_fps <= 0:  # Fallback if FPS detection fails
                self.video_fps = 30
                print("Warning: Could not detect video FPS, using default 30fps")
        else:
            self.width = 640
            self.height = 480
            self.video_fps = 30
        
        print(f"Video dimensions: {self.width}x{self.height}")
        print(f"Video FPS: {self.video_fps}")
        
        # Load YOLOv8 models
        self.player_model = YOLO('yolov8n.pt')
        self.ball_model = YOLO('best.pt')
        self.rim_model = YOLO('best.pt')
        self.pose_model = YOLO('yolo11n-pose.pt')  # Add pose model
        
        # Initialize trackers
        self.hoop_tracker = HoopTracker(self.video_fps)  # Pass actual FPS
        self.ball_trajectory = []
        self.shot_trajectory = []  # Separate trajectory for active shot
        
        # Shot tracking variables
        self.potential_shot = False
        self.shot_start_frame = 0
        self.last_shot_frame = 0
        self.total_shots = 0
        self.made_shots = 0
        self.missed_shots = 0
        
        # Enhanced shot detection variables
        self.upward_motion_frames = 0  # Count consecutive upward motion frames
        self.min_upward_frames = max(2, self.video_fps // 10)  # Adaptive based on FPS
        self.last_y_positions = []     # Store last N y-positions for smoothing
        self.max_y_positions = max(3, self.video_fps // 6)  # Adaptive based on FPS
        
        # Timeout values in seconds, will be converted to frames based on FPS
        self.shot_timeout_seconds = 1.5     # Base timeout of 1.5 seconds
        self.extension_seconds = 0.5        # Extension of 0.5 seconds when near hoop
        
        # Convert seconds to frames based on actual FPS
        self.shot_timeout_frames = int(self.shot_timeout_seconds * self.video_fps)
        self.shot_extension_frames = int(self.extension_seconds * self.video_fps)
        
        print(f"Shot detection parameters for {self.video_fps}fps video:")
        print(f"- Minimum upward motion frames: {self.min_upward_frames}")
        print(f"- Position history size: {self.max_y_positions}")
        print(f"- Base timeout: {self.shot_timeout_frames} frames ({self.shot_timeout_seconds}s)")
        print(f"- Extension when near hoop: {self.shot_extension_frames} frames ({self.extension_seconds}s)")
        
        # Pose-based shot detection
        self.pose_sequence = deque(maxlen=30)  # Store last 30 frames of pose data
        self.min_frames_for_shot = 5
        self.max_frames_for_shot = 60
        
        # Threading components
        self.frame_queue = Queue(maxsize=30)
        self.processed_frame_queue = Queue(maxsize=30)
        self.is_running = False
        self.processing_thread = None

    def is_ball_near_hoop(self, ball_pos):
        """Check if ball is near the hoop area."""
        if not self.hoop_tracker.hoop_found or not ball_pos:
            return False
            
        hoop_x, hoop_y = self.hoop_tracker.hoop_center
        ball_x, ball_y = ball_pos
        
        # Define "near" as within 1.5x hoop width horizontally and vertically
        hoop_width = self.hoop_tracker.hoop_position[2]
        distance_threshold = hoop_width * 1.5
        
        dx = abs(ball_x - hoop_x)
        dy = abs(ball_y - hoop_y)
        
        return dx < distance_threshold and dy < distance_threshold

    def detect_upward_motion(self, current_pos):
        """Enhanced upward motion detection with smoothing."""
        if not current_pos:
            self.upward_motion_frames = 0
            return False
            
        # Add current y position to history
        self.last_y_positions.append(current_pos[1])
        if len(self.last_y_positions) > self.max_y_positions:
            self.last_y_positions.pop(0)
            
        # Need at least 3 points for reliable detection
        if len(self.last_y_positions) < 3:
            return False
            
        # Calculate average velocity over last few frames
        y_velocities = [self.last_y_positions[i] - self.last_y_positions[i-1] 
                       for i in range(1, len(self.last_y_positions))]
        avg_velocity = sum(y_velocities) / len(y_velocities)
        
        # Check if moving upward (negative velocity in screen coordinates)
        if avg_velocity < -3:  # Threshold for upward motion
            self.upward_motion_frames += 1
        else:
            self.upward_motion_frames = 0
            
        return self.upward_motion_frames >= self.min_upward_frames

    def preprocess_for_ball_detection(self, frame):
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define orange-brown range for basketball
        lower_ball = np.array([0, 50, 50])
        upper_ball = np.array([30, 255, 255])
        
        # Create mask for basketball colors
        mask = cv2.inRange(hsv, lower_ball, upper_ball)
        
        # Apply morphological operations
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        return frame, mask

    def detect_circles(self, mask):
        circles = cv2.HoughCircles(
            mask,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=40
        )
        
        detections = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (int(i[0]), int(i[1]))
                radius = int(i[2])
                
                detections.append({
                    'bbox': [
                        max(0, center[0] - radius),
                        max(0, center[1] - radius),
                        center[0] + radius,
                        center[1] + radius
                    ],
                    'center': center,
                    'radius': radius
                })
        
        return detections

    def detect_ball(self, frame):
        # Run YOLOv8 inference for basketball detection
        results = self.ball_model(frame, classes=[0])
        yolo_detections = []
        
        # Process YOLOv8 results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                if conf > 0.3:
                    yolo_detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'conf': conf,
                        'center': ((x1 + x2) // 2, (y1 + y2) // 2)
                    })

        # Color and circle-based detection
        processed_frame, color_mask = self.preprocess_for_ball_detection(frame)
        circle_detections = self.detect_circles(color_mask)
        
        # Combine detections
        all_detections = []
        
        # Add YOLO detections
        for det in yolo_detections:
            det['method'] = 'yolo'
            all_detections.append(det)
        
        # Add circle detections with confidence boost if near YOLO detection
        for circle in circle_detections:
            circle['method'] = 'circle'
            circle['conf'] = 0.3  # Base confidence for circle detection
            
            # Boost confidence if near YOLO detection
            for yolo_det in yolo_detections:
                dist = np.sqrt((circle['center'][0] - yolo_det['center'][0])**2 +
                             (circle['center'][1] - yolo_det['center'][1])**2)
                if dist < 50:  # If circle is near YOLO detection
                    circle['conf'] = min(0.9, circle['conf'] + yolo_det['conf'])
                    
            all_detections.append(circle)
        
        return all_detections

    def detect_players(self, frame):
        results = self.player_model(frame, classes=[0])  # class 0 is person
        detections = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                if conf > 0.5:  # Filter low confidence detections
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'conf': conf,
                        'center': ((x1 + x2) // 2, (y1 + y2) // 2)
                    })
        
        return detections
        
    def detect_hoop(self, frame):
        # Run YOLOv8 inference for rim detection (class 1 is rim)
        results = self.rim_model(frame, classes=[1])
        detections = []
        
        # Process YOLOv8 results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                if conf > 0.3:  # Lower threshold for rim detection
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'conf': conf,
                        'center': ((x1 + x2) // 2, (y1 + y2) // 2)
                    })
        
        return detections

    def detect_poses(self, frame):
        """Detect poses in the frame using YOLOv8-pose."""
        results = self.pose_model(frame, conf=0.5)
        return results[0] if results else None

    def draw_pose_skeleton(self, frame, pose_result):
        """Draw the pose skeleton on the frame."""
        if pose_result and pose_result.keypoints is not None:
            keypoints = pose_result.keypoints.data
            for kpts in keypoints:
                # Draw keypoints
                for kpt in kpts:
                    x, y = int(kpt[0]), int(kpt[1])
                    conf = kpt[2]
                    if conf > 0.5:  # Only draw high confidence keypoints
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                
                # Draw skeleton lines
                skeleton = [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
                for line in skeleton:
                    pt1 = (int(kpts[line[0]-1][0]), int(kpts[line[0]-1][1]))
                    pt2 = (int(kpts[line[1]-1][0]), int(kpts[line[1]-1][1]))
                    if kpts[line[0]-1][2] > 0.5 and kpts[line[1]-1][2] > 0.5:  # Only draw if both points are confident
                        cv2.line(frame, pt1, pt2, (0, 255, 0), 1)
        return frame

    def extract_keypoints(self, pose_result):
        """Extract keypoints from pose result into a dictionary."""
        if pose_result and pose_result.keypoints is not None and len(pose_result.keypoints.data) > 0:
            keypoints = pose_result.keypoints.data[0]  # Get first person's keypoints
            return {i: (int(kpt[0]), int(kpt[1])) if kpt[2] > 0.5 else None 
                   for i, kpt in enumerate(keypoints)}
        return None

    def get_keypoint_y(self, keypoints, index, default=float('inf')):
        """Safely get Y coordinate of a keypoint."""
        if keypoints and index in keypoints and keypoints[index] is not None:
            return keypoints[index][1]
        return default

    def detect_shot_from_pose(self, pose_sequence):
        """Determine if the given sequence of pose keypoints indicates a basketball shot attempt."""
        if len(pose_sequence) < self.min_frames_for_shot:
            return False

        # Filter out None values from sequence
        valid_frames = [frame for frame in pose_sequence if frame is not None]
        if len(valid_frames) < self.min_frames_for_shot:
            return False

        # Compute joint angles for each frame
        right_elbow_angles = []
        right_knee_angles = []
        left_elbow_angles = []
        wrist_y = []

        for frame_keypoints in valid_frames:
            # Get required keypoints for right arm
            r_shoulder = frame_keypoints.get(RIGHT_SHOULDER)
            r_elbow = frame_keypoints.get(RIGHT_ELBOW)
            r_wrist = frame_keypoints.get(RIGHT_WRIST)
            
            # Get required keypoints for right leg
            r_hip = frame_keypoints.get(RIGHT_HIP)
            r_knee = frame_keypoints.get(RIGHT_KNEE)
            r_ankle = frame_keypoints.get(RIGHT_ANKLE)
            
            # Get required keypoints for left arm
            l_shoulder = frame_keypoints.get(LEFT_SHOULDER)
            l_elbow = frame_keypoints.get(LEFT_ELBOW)
            l_wrist = frame_keypoints.get(LEFT_WRIST)

            # Calculate angles only if all required keypoints are present
            if all(p is not None for p in [r_shoulder, r_elbow, r_wrist]):
                right_elbow_angles.append(joint_angle(r_shoulder, r_elbow, r_wrist))
            
            if all(p is not None for p in [r_hip, r_knee, r_ankle]):
                right_knee_angles.append(joint_angle(r_hip, r_knee, r_ankle))
            
            if all(p is not None for p in [l_shoulder, l_elbow, l_wrist]):
                left_elbow_angles.append(joint_angle(l_shoulder, l_elbow, l_wrist))
            
            if r_wrist is not None:
                wrist_y.append(r_wrist[1])

        # Check if we have enough data to analyze
        if not (right_elbow_angles and right_knee_angles and wrist_y):
            return False

        # Analyze knee movement
        min_knee_angle = min(right_knee_angles)
        min_knee_index = right_knee_angles.index(min_knee_angle)
        end_knee_angle = right_knee_angles[-1]

        knee_bent_enough = min_knee_angle < 140
        knee_extended_enough = end_knee_angle > min_knee_angle + 20
        valid_knee_sequence = (knee_bent_enough and knee_extended_enough and 
                             0 < min_knee_index < len(valid_frames) - 1)

        # Analyze elbow movement
        start_elbow_angle = right_elbow_angles[0]
        end_elbow_angle = right_elbow_angles[-1]
        elbow_started_bent = start_elbow_angle < 140
        elbow_fully_extended = end_elbow_angle > 150
        elbow_angle_increased = end_elbow_angle - start_elbow_angle > 30

        # Analyze wrist movement
        wrist_velocities = [wrist_y[i] - wrist_y[i-1] for i in range(1, len(wrist_y))]
        upward_motion = any(v < -5.0 for v in wrist_velocities)

        # Check final positions
        final_keypoints = valid_frames[-1] if valid_frames else None
        if final_keypoints:
            final_wrist_y = self.get_keypoint_y(final_keypoints, RIGHT_WRIST)
            final_shoulder_y = self.get_keypoint_y(final_keypoints, RIGHT_SHOULDER)
            wrist_above_shoulder = final_wrist_y < final_shoulder_y
        else:
            wrist_above_shoulder = False

        # Check guide hand
        right_arm_extended = elbow_fully_extended
        left_arm_extended = left_elbow_angles and left_elbow_angles[-1] > 150
        guide_hand_support = not (right_arm_extended and left_arm_extended)

        # Duration check
        duration_ok = self.min_frames_for_shot <= len(valid_frames) <= self.max_frames_for_shot

        # Final decision
        return (elbow_started_bent and elbow_fully_extended and elbow_angle_increased and
                upward_motion and wrist_above_shoulder and valid_knee_sequence and
                guide_hand_support and duration_ok)

    def process_frame(self, frame, frame_count):
        """Process a single frame and detect shots."""
        if frame is None:
            return None
            
        # Get ball detections
        ball_detections = self.detect_ball(frame)
        
        # Get hoop detections
        hoop_detections = self.detect_hoop(frame)
        
        # Get player detections and draw boxes
        player_detections = self.detect_players(frame)
        for player in player_detections:
            x1, y1, x2, y2 = player['bbox']
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        
        # Get and draw pose detections
        pose_results = self.detect_poses(frame)
        frame = self.draw_pose_skeleton(frame, pose_results)
        
        # Extract and store pose keypoints for shot detection
        keypoints = self.extract_keypoints(pose_results)
        if keypoints:
            self.pose_sequence.append(keypoints)
            
            # Only attempt shot detection if we're not already tracking a shot
            if not self.potential_shot and len(self.pose_sequence) >= self.min_frames_for_shot:
                try:
                    if self.detect_shot_from_pose(list(self.pose_sequence)):
                        print(f"\nFrame {frame_count}: Starting shot tracking - Detected shot attempt from pose analysis")
                        self.potential_shot = True
                        self.shot_trajectory = []
                        self.shot_start_frame = frame_count
                except Exception as e:
                    print(f"Error in pose-based shot detection: {e}")
                    # Continue with normal processing even if pose detection fails
        
        # Update trackers
        self.hoop_tracker.update_hoop(hoop_detections)
        
        # Update ball position if detected
        current_ball_pos = None
        current_ball_bbox = None
        if ball_detections:
            # Find the ball with highest confidence
            best_ball = max(ball_detections, key=lambda x: x['conf'])
            x1, y1, x2, y2 = best_ball['bbox']
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            current_ball_pos = (center_x, center_y)
            current_ball_bbox = (int(x1), int(y1), int(x2), int(y2))
            
            # Draw ball boundary box in orange
            cv2.rectangle(frame, 
                        (int(x1), int(y1)), 
                        (int(x2), int(y2)), 
                        (0, 165, 255), 2)  # Orange in BGR
            
            # Update main trajectory
            self.ball_trajectory.append(current_ball_pos)
            if len(self.ball_trajectory) > 30:  # Keep last 30 points
                self.ball_trajectory.pop(0)
        
        # Draw ball trajectory
        if len(self.ball_trajectory) >= 2:
            # Draw main trajectory in yellow
            for i in range(1, len(self.ball_trajectory)):
                pt1 = (int(self.ball_trajectory[i-1][0]), int(self.ball_trajectory[i-1][1]))
                pt2 = (int(self.ball_trajectory[i][0]), int(self.ball_trajectory[i][1]))
                cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
            
            # Draw shot trajectory in red if tracking a shot
            if self.potential_shot and len(self.shot_trajectory) >= 2:
                for i in range(1, len(self.shot_trajectory)):
                    pt1 = (int(self.shot_trajectory[i-1][0]), int(self.shot_trajectory[i-1][1]))
                    pt2 = (int(self.shot_trajectory[i][0]), int(self.shot_trajectory[i][1]))
                    cv2.line(frame, pt1, pt2, (0, 0, 255), 2)
        
        # Draw boxes for visualization
        if self.hoop_tracker.hoop_found:
            frame = self.hoop_tracker.draw_detection_boxes(frame)
        
        # Shot detection logic
        if self.hoop_tracker.hoop_found and len(self.ball_trajectory) >= 3:
            if not self.potential_shot:
                # Enhanced shot detection using smoothed upward motion
                if self.detect_upward_motion(current_ball_pos):
                    print(f"\nFrame {frame_count}: Starting shot tracking - Detected consistent upward motion")
                    self.potential_shot = True
                    self.shot_trajectory = []
                    self.shot_start_frame = frame_count
            
            # If tracking a shot and ball is detected, update shot trajectory
            if self.potential_shot and current_ball_pos:
                self.shot_trajectory.append(current_ball_pos)
                
                # Check for shot outcome
                if len(self.shot_trajectory) >= 3:
                    shot_info = {
                        'points': self.shot_trajectory,
                        'current_bbox': current_ball_bbox
                    }
                    outcome = self.hoop_tracker.detect_shot_outcome(shot_info)
                    
                    if outcome:
                        print(f"Frame {frame_count}: Shot outcome - {outcome}")
                        self.total_shots += 1
                        if outcome == 'made':
                            self.made_shots += 1
                        else:
                            self.missed_shots += 1
                        
                        # Reset for next shot
                        self.potential_shot = False
                        self.shot_trajectory = []
                        self.upward_motion_frames = 0
                        self.last_y_positions = []
                        print(f"Total: {self.total_shots}, Made: {self.made_shots}, Missed: {self.missed_shots}")
                    
                    # Dynamic timeout based on ball position
                    timeout_frames = self.shot_timeout_frames
                    if self.is_ball_near_hoop(current_ball_pos):
                        timeout_frames += self.shot_extension_frames
                        
                    if frame_count - self.shot_start_frame > timeout_frames:
                        print(f"Frame {frame_count}: Shot tracking timeout")
                        # Final outcome check with timeout flag
                        final_outcome = self.hoop_tracker.detect_shot_outcome(shot_info, True)
                        if final_outcome:
                            print(f"Frame {frame_count}: Last-chance shot outcome - {final_outcome}")
                            self.total_shots += 1
                            if final_outcome == 'made':
                                self.made_shots += 1
                            else:
                                self.missed_shots += 1
                            print(f"Total: {self.total_shots}, Made: {self.made_shots}, Missed: {self.missed_shots}")
                        
                        # Reset all tracking variables
                        self.potential_shot = False
                        self.shot_trajectory = []
                        self.upward_motion_frames = 0
                        self.last_y_positions = []
                        self.hoop_tracker.reset_box_states()
        
        # Draw stats overlay
        frame = self.draw_stats(frame)
        
        return frame

    def draw_stats(self, frame):
        """Draw stats in the bottom-left corner of the frame."""
        # Define the position and size of the stats box
        box_x, box_y = 10, self.height - 90  # Bottom-left corner
        box_width, box_height = 200, 80

        # Draw white background rectangle
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (255, 255, 255), -1)

        # Draw stats text in black
        cv2.putText(frame, f"Total Shots: {self.total_shots}", (box_x + 10, box_y + 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(frame, f"Made: {self.made_shots}", (box_x + 10, box_y + 45),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(frame, f"Missed: {self.missed_shots}", (box_x + 10, box_y + 65),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 2)

        # Draw percentage if we have shots
        if self.total_shots > 0:
            percentage = (self.made_shots / self.total_shots) * 100
            cv2.putText(frame, f"{percentage:.1f}%", (box_x + 120, box_y + 45),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 2)

        return frame

    def process_video_file(self):
        """Process a video file and save the output."""
        if not self.cap or not self.output_path:
            print("Error: Video capture or output path not set")
            return

        # Get video properties
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer with high quality settings
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.video_fps,
            (self.width, self.height),
            True
        )

        frame_count = 0
        start_time = time.time()
        last_time = start_time
        fps_update_interval = 1.0  # Update FPS display every second
        
        pbar = tqdm(total=total_frames, desc="Processing video")

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Process frame
            frame_start_time = time.time()
            processed_frame = self.process_frame(frame, frame_count)
            frame_process_time = time.time() - frame_start_time
            
            if processed_frame is not None:
                out.write(processed_frame)
            
            # Update progress and FPS statistics
            frame_count += 1
            pbar.update(1)
            
            current_time = time.time()
            if current_time - last_time >= fps_update_interval:
                elapsed = current_time - start_time
                avg_fps = frame_count / elapsed
                current_fps = 1.0 / frame_process_time
                pbar.set_description(f"Processing video [Avg: {avg_fps:.1f} fps, Current: {current_fps:.1f} fps]")
                last_time = current_time

        # Final timing statistics
        total_time = time.time() - start_time
        print(f"\nProcessing complete:")
        print(f"Total frames: {frame_count}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average FPS: {frame_count/total_time:.1f}")
        print(f"Original video FPS: {self.video_fps}")
        print(f"Processing speed: {(frame_count/total_time)/self.video_fps:.1f}x realtime")

        # Clean up
        pbar.close()
        self.cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        self.print_stats()

    def process_realtime(self):
        """Process video in real-time from camera or video file."""
        if not self.cap:
            print("Opening default camera...")
            self.cap = cv2.VideoCapture(0)  # Use default camera
            if not self.cap.isOpened():
                print("Error: Could not open camera")
                return

        frame_count = 0
        start_time = time.time()
        last_time = start_time
        fps_update_interval = 1.0  # Update FPS display every second
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                if self.cap.get(cv2.CAP_PROP_POS_FRAMES) == self.cap.get(cv2.CAP_PROP_FRAME_COUNT):
                    # If we've reached the end of the video file, loop back to start
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            # Process frame
            frame_start_time = time.time()
            processed_frame = self.process_frame(frame, frame_count)
            frame_process_time = time.time() - frame_start_time
            
            if processed_frame is not None:
                # Update FPS calculation
                current_time = time.time()
                if current_time - last_time >= fps_update_interval:
                    elapsed = current_time - start_time
                    avg_fps = frame_count / elapsed
                    current_fps = 1.0 / frame_process_time
                    print(f"\rProcessing: Avg FPS: {avg_fps:.1f}, Current FPS: {current_fps:.1f}", end="")
                    last_time = current_time
                
                # Display the frame
                cv2.imshow('Basketball Shot Tracking', processed_frame)
                
                # Break if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1

        # Final timing statistics
        total_time = time.time() - start_time
        print(f"\nProcessing complete:")
        print(f"Total frames: {frame_count}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average FPS: {frame_count/total_time:.1f}")
        print(f"Original video FPS: {self.video_fps}")
        print(f"Processing speed: {(frame_count/total_time)/self.video_fps:.1f}x realtime")

        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        self.print_stats()

    def print_stats(self):
        """Print final statistics."""
        print(f"\nFinal Statistics:")
        print(f"Total Shots: {self.total_shots}")
        print(f"Made Shots: {self.made_shots}")
        print(f"Missed Shots: {self.missed_shots}")
        print(f"Shooting Percentage: {(self.made_shots / self.total_shots * 100) if self.total_shots > 0 else 0:.1f}%")

if __name__ == '__main__':
    # DO NOT CHANGE
    parser = argparse.ArgumentParser(description='Basketball Shot Tracking')
    parser.add_argument('--mode', type=str, choices=['realtime', 'video'], 
                      help='Processing mode: realtime for live processing, video for file processing')
    parser.add_argument('--input', type=str, help='Input video file path (optional for realtime mode)')
    parser.add_argument('--output', type=str, help='Output video file path (required for video mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'video':
        if not args.input or not args.output:
            parser.error("Video mode requires both input and output file paths")
        tracker = BasketballTracker(args.input, args.output)
        tracker.process_video_file()
    elif args.mode == 'realtime':
        tracker = BasketballTracker(args.input if args.input else None)
        tracker.process_realtime()
    else:
        parser.error("Must specify processing mode (realtime or video)")
