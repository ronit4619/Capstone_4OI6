import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque
import argparse
from math import acos, degrees, sqrt

def joint_angle(A, B, C):
    if any(p is None for p in [A, B, C]):
        return 0.0
    BAx, BAy = A[0] - B[0], A[1] - B[1]
    BCx, BCy = C[0] - B[0], C[1] - B[1]
    dot = BAx * BCx + BAy * BCy
    magBA = sqrt(BAx**2 + BAy**2)
    magBC = sqrt(BCx**2 + BCy**2)
    if magBA == 0 or magBC == 0:
        return 0.0
    cos_angle = max(-1.0, min(1.0, dot / (magBA * magBC)))
    angle_rad = acos(cos_angle)
    return degrees(angle_rad)

class HoopTracker:
    def __init__(self, fps):
        self.fps = fps
        self.hoop_found = False
        self.hoop_position = None  # (x, y, w, h)
        self.hoop_center = None    # (x, y)
        self.entry_box_touched = False
        self.make_box_touched = False
        self.current_frame = 0
        self.max_time_between_boxes = 1.0  # seconds
        self.max_frames_between_boxes = int(self.max_time_between_boxes * self.fps)
        
    def reset_box_states(self):
        self.entry_box_touched = False
        self.make_box_touched = False

    def bboxes_overlap(self, bbox1, bbox2):
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        overlap_x = (x1_min <= x2_max) and (x2_min <= x1_max)
        overlap_y = (y1_min <= y2_max) and (y2_min <= y1_max)
        return overlap_x and overlap_y

    def draw_detection_boxes(self, frame):
        if not self.hoop_found:
            return frame
        hoop_width = self.hoop_position[2]
        hoop_height = self.hoop_position[3]
        hoop_center_x = self.hoop_center[0]
        hoop_center_y = self.hoop_center[1]
        # Draw the hoop boundary (yellow)
        hoop_box = (
            int(hoop_center_x - hoop_width/2),
            int(hoop_center_y - hoop_height/2),
            int(hoop_center_x + hoop_width/2),
            int(hoop_center_y + hoop_height/2)
        )
        cv2.rectangle(frame, (hoop_box[0], hoop_box[1]), (hoop_box[2], hoop_box[3]), (0, 255, 255), 2)
        # Define entry and make boxes
        entry_box_width = hoop_width * 0.5
        entry_box = (
            int(hoop_center_x - entry_box_width/2),
            int(hoop_center_y - entry_box_width * 2),
            int(hoop_center_x + entry_box_width/2),
            int(hoop_center_y - entry_box_width)
        )
        make_box_width = hoop_width
        make_box_height = hoop_height * 0.5
        make_box = (
            int(hoop_center_x - make_box_width/2),
            int(hoop_center_y),
            int(hoop_center_x + make_box_width/2),
            int(hoop_center_y + make_box_height)
        )
        entry_color = (0, 255, 0) if self.entry_box_touched else (255, 255, 255)
        cv2.rectangle(frame, (entry_box[0], entry_box[1]), (entry_box[2], entry_box[3]), entry_color, 2)
        make_color = (0, 255, 0) if self.make_box_touched else (255, 255, 255)
        cv2.rectangle(frame, (make_box[0], make_box[1]), (make_box[2], make_box[3]), make_color, 2)
        return frame

    def detect_shot_outcome(self, trajectory_info, is_timeout=False):
        if not trajectory_info or not self.hoop_found:
            return None
        points = trajectory_info['points']
        ball_bbox = trajectory_info.get('current_bbox')
        if len(points) < 2 or not ball_bbox:
            return None

        hoop_width = self.hoop_position[2]
        hoop_height = self.hoop_position[3]
        hoop_center_x = self.hoop_center[0]
        hoop_center_y = self.hoop_center[1]
        # Define entry and make boxes based on hoop dimensions
        entry_box_width = hoop_width * 0.5
        entry_box = (
            int(hoop_center_x - entry_box_width/2),
            int(hoop_center_y - entry_box_width),
            int(hoop_center_x + entry_box_width/2),
            int(hoop_center_y)
        )
        make_box_width = hoop_width
        make_box_height = hoop_height * 0.5
        make_box = (
            int(hoop_center_x - make_box_width/2),
            int(hoop_center_y),
            int(hoop_center_x + make_box_width/2),
            int(hoop_center_y + make_box_height)
        )
        # Define a larger hoop box to track ball leaving
        hoop_box = (
            int(hoop_center_x - hoop_width),
            int(hoop_center_y - hoop_height),
            int(hoop_center_x + hoop_width),
            int(hoop_center_y + hoop_height)
        )
        # If the make box is hit before the entry box (and not during timeout), reset states
        if not self.entry_box_touched and self.bboxes_overlap(ball_bbox, make_box) and not is_timeout:
            self.reset_box_states()
            return None
        if not self.entry_box_touched and self.bboxes_overlap(ball_bbox, entry_box):
            print("Ball touched entry box")
            self.entry_box_touched = True
        if is_timeout and self.entry_box_touched:
            if self.bboxes_overlap(ball_bbox, make_box):
                print("Ball touched make box during timeout check - MADE")
                return 'made'
            elif not self.bboxes_overlap(ball_bbox, hoop_box):
                print("Ball left hoop area during timeout check - MISS")
                return 'missed'
            return None
        if self.entry_box_touched and not self.make_box_touched and not self.bboxes_overlap(ball_bbox, hoop_box):
            print("Ball left hoop area - MISS")
            self.reset_box_states()
            return 'missed'
        if self.entry_box_touched and not self.make_box_touched and self.bboxes_overlap(ball_bbox, make_box):
            print("Ball touched make box - MADE")
            self.make_box_touched = True
            self.reset_box_states()
            return 'made'
        self.current_frame += 1
        return None

    def update_hoop(self, detections):
        if not self.hoop_found and detections:
            best_rim = max(detections, key=lambda x: x['conf'])
            x1, y1, x2, y2 = best_rim['bbox']
            self.hoop_position = (x1, y1, x2 - x1, y2 - y1)
            self.hoop_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            self.hoop_found = True

class BasketballTracker:
    def __init__(self, video_path=None, output_path="/output"):
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
        else:
            self.cap = None
        self.output_path = output_path
        
        if self.cap:
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            if self.video_fps <= 0:
                self.video_fps = 30
                print("Warning: Could not detect video FPS, using default 30fps")
        else:
            self.width = 640
            self.height = 480
            self.video_fps = 30
        
        print(f"Video dimensions: {self.width}x{self.height}")
        print(f"Video FPS: {self.video_fps}")
        
        # Load only the ball and rim models
        self.ball_model = YOLO('best.pt')
        self.rim_model = YOLO('best.pt')
        
        self.hoop_tracker = HoopTracker(self.video_fps)
        self.ball_trajectory = []
        self.shot_trajectory = []
        
        self.potential_shot = False
        self.shot_start_frame = 0
        self.total_shots = 0
        self.made_shots = 0
        self.missed_shots = 0
        
        self.upward_motion_frames = 0
        self.min_upward_frames = max(2, self.video_fps // 10)
        self.last_y_positions = []
        self.max_y_positions = max(3, self.video_fps // 6)
        
        self.shot_timeout_seconds = 1.5
        self.extension_seconds = 0.5
        self.shot_timeout_frames = int(self.shot_timeout_seconds * self.video_fps)
        self.shot_extension_frames = int(self.extension_seconds * self.video_fps)
        
        print(f"Shot detection parameters for {self.video_fps}fps video:")
        print(f"- Minimum upward motion frames: {self.min_upward_frames}")
        print(f"- Position history size: {self.max_y_positions}")
        print(f"- Base timeout: {self.shot_timeout_frames} frames ({self.shot_timeout_seconds}s)")
        print(f"- Extension when near hoop: {self.shot_extension_frames} frames ({self.extension_seconds}s)")
        
    def detect_upward_motion(self, current_pos):
        if not current_pos:
            self.upward_motion_frames = 0
            return False
        self.last_y_positions.append(current_pos[1])
        if len(self.last_y_positions) > self.max_y_positions:
            self.last_y_positions.pop(0)
        if len(self.last_y_positions) < 3:
            return False
        y_velocities = [self.last_y_positions[i] - self.last_y_positions[i-1]
                        for i in range(1, len(self.last_y_positions))]
        avg_velocity = sum(y_velocities) / len(y_velocities)
        if avg_velocity < -3:  # upward motion in screen coordinates
            self.upward_motion_frames += 1
        else:
            self.upward_motion_frames = 0
        return self.upward_motion_frames >= self.min_upward_frames

    def preprocess_for_ball_detection(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_ball = np.array([0, 50, 50])
        upper_ball = np.array([30, 255, 255])
        mask = cv2.inRange(hsv, lower_ball, upper_ball)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        return frame, mask

    def detect_circles(self, mask):
        circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                                   param1=50, param2=30, minRadius=10, maxRadius=40)
        detections = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (int(i[0]), int(i[1]))
                radius = int(i[2])
                detections.append({
                    'bbox': [max(0, center[0] - radius),
                             max(0, center[1] - radius),
                             center[0] + radius,
                             center[1] + radius],
                    'center': center,
                    'radius': radius
                })
        return detections

    def detect_ball(self, frame):
        results = self.ball_model(frame, classes=[0])
        yolo_detections = []
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
        _, color_mask = self.preprocess_for_ball_detection(frame)
        circle_detections = self.detect_circles(color_mask)
        all_detections = []
        for det in yolo_detections:
            det['method'] = 'yolo'
            all_detections.append(det)
        for circle in circle_detections:
            circle['method'] = 'circle'
            circle['conf'] = 0.3
            for yolo_det in yolo_detections:
                dist = np.sqrt((circle['center'][0] - yolo_det['center'][0]) ** 2 +
                               (circle['center'][1] - yolo_det['center'][1]) ** 2)
                if dist < 50:
                    circle['conf'] = min(0.9, circle['conf'] + yolo_det['conf'])
            all_detections.append(circle)
        return all_detections

    def detect_hoop(self, frame):
        results = self.rim_model(frame, classes=[1])
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                if conf > 0.3:
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'conf': conf,
                        'center': ((x1 + x2) // 2, (y1 + y2) // 2)
                    })
        return detections

    def is_ball_near_hoop(self, ball_pos):
        if not self.hoop_tracker.hoop_found or not ball_pos:
            return False
        hoop_x, hoop_y = self.hoop_tracker.hoop_center
        ball_x, ball_y = ball_pos
        hoop_width = self.hoop_tracker.hoop_position[2]
        distance_threshold = hoop_width * 1.5
        return abs(ball_x - hoop_x) < distance_threshold and abs(ball_y - hoop_y) < distance_threshold

    def process_frame(self, frame, frame_count):
        if frame is None:
            return None
        # Detect ball and hoop
        ball_detections = self.detect_ball(frame)
        hoop_detections = self.detect_hoop(frame)
        self.hoop_tracker.update_hoop(hoop_detections)
        
        # Update ball trajectory if a ball is detected
        current_ball_pos = None
        current_ball_bbox = None
        if ball_detections:
            best_ball = max(ball_detections, key=lambda x: x['conf'])
            x1, y1, x2, y2 = best_ball['bbox']
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            current_ball_pos = (center_x, center_y)
            current_ball_bbox = (int(x1), int(y1), int(x2), int(y2))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 2)
            self.ball_trajectory.append(current_ball_pos)
            if len(self.ball_trajectory) > 30:
                self.ball_trajectory.pop(0)
                
        # Draw ball trajectory
        if len(self.ball_trajectory) >= 2:
            for i in range(1, len(self.ball_trajectory)):
                pt1 = (int(self.ball_trajectory[i - 1][0]), int(self.ball_trajectory[i - 1][1]))
                pt2 = (int(self.ball_trajectory[i][0]), int(self.ball_trajectory[i][1]))
                cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
            if self.potential_shot and len(self.shot_trajectory) >= 2:
                for i in range(1, len(self.shot_trajectory)):
                    pt1 = (int(self.shot_trajectory[i - 1][0]), int(self.shot_trajectory[i - 1][1]))
                    pt2 = (int(self.shot_trajectory[i][0]), int(self.shot_trajectory[i][1]))
                    cv2.line(frame, pt1, pt2, (0, 0, 255), 2)
                    
        # Draw hoop detection boxes
        if self.hoop_tracker.hoop_found:
            frame = self.hoop_tracker.draw_detection_boxes(frame)
            
        # Shot detection logic based on upward ball motion
        if self.hoop_tracker.hoop_found and len(self.ball_trajectory) >= 3:
            if not self.potential_shot:
                if self.detect_upward_motion(current_ball_pos):
                    print(f"\nFrame {frame_count}: Starting shot tracking - Detected consistent upward motion")
                    self.potential_shot = True
                    self.shot_trajectory = []
                    self.shot_start_frame = frame_count
            if self.potential_shot and current_ball_pos:
                self.shot_trajectory.append(current_ball_pos)
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
                        self.potential_shot = False
                        self.shot_trajectory = []
                        self.upward_motion_frames = 0
                        self.last_y_positions = []
                        print(f"Total: {self.total_shots}, Made: {self.made_shots}, Missed: {self.missed_shots}")
                    timeout_frames = self.shot_timeout_frames
                    if self.is_ball_near_hoop(current_ball_pos):
                        timeout_frames += self.shot_extension_frames
                    if frame_count - self.shot_start_frame > timeout_frames:
                        print(f"Frame {frame_count}: Shot tracking timeout")
                        final_outcome = self.hoop_tracker.detect_shot_outcome(shot_info, True)
                        if final_outcome:
                            print(f"Frame {frame_count}: Last-chance shot outcome - {final_outcome}")
                            self.total_shots += 1
                            if final_outcome == 'made':
                                self.made_shots += 1
                            else:
                                self.missed_shots += 1
                            print(f"Total: {self.total_shots}, Made: {self.made_shots}, Missed: {self.missed_shots}")
                        self.potential_shot = False
                        self.shot_trajectory = []
                        self.upward_motion_frames = 0
                        self.last_y_positions = []
                        self.hoop_tracker.reset_box_states()
                        
        frame = self.draw_stats(frame)
        return frame

    def draw_stats(self, frame):
        cv2.rectangle(frame, (10, 10), (200, 90), (0, 0, 0), -1)
        cv2.putText(frame, f"Total Shots: {self.total_shots}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Made: {self.made_shots}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Missed: {self.missed_shots}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if self.total_shots > 0:
            percentage = (self.made_shots / self.total_shots) * 100
            cv2.putText(frame, f"{percentage:.1f}%", (120, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return frame

    def process_video_file(self):
        if not self.cap or not self.output_path:
            print("Error: Video capture or output path not set")
            return
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(self.output_path, fourcc, self.video_fps, (self.width, self.height), True)
        frame_count = 0
        start_time = time.time()
        last_time = start_time
        fps_update_interval = 1.0
        from tqdm import tqdm
        pbar = tqdm(total=total_frames, desc="Processing video")
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            frame_start_time = time.time()
            processed_frame = self.process_frame(frame, frame_count)
            if processed_frame is not None:
                out.write(processed_frame)
            frame_count += 1
            pbar.update(1)
            current_time = time.time()
            if current_time - last_time >= fps_update_interval:
                elapsed = current_time - start_time
                avg_fps = frame_count / elapsed
                current_fps = 1.0 / (time.time() - frame_start_time)
                pbar.set_description(f"Processing video [Avg: {avg_fps:.1f} fps, Current: {current_fps:.1f} fps]")
                last_time = current_time
        total_time = time.time() - start_time
        print(f"\nProcessing complete:")
        print(f"Total frames: {frame_count}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average FPS: {frame_count / total_time:.1f}")
        print(f"Original video FPS: {self.video_fps}")
        print(f"Processing speed: {((frame_count / total_time) / self.video_fps):.1f}x realtime")
        pbar.close()
        self.cap.release()
        out.release()
        cv2.destroyAllWindows()
        self.print_stats()

    def process_realtime(self):
        if not self.cap:
            print("Opening default camera...")
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Error: Could not open camera")
                return
        frame_count = 0
        start_time = time.time()
        last_time = start_time
        fps_update_interval = 1.0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                if self.cap.get(cv2.CAP_PROP_POS_FRAMES) == self.cap.get(cv2.CAP_PROP_FRAME_COUNT):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            frame_start_time = time.time()
            processed_frame = self.process_frame(frame, frame_count)
            if processed_frame is not None:
                current_time = time.time()
                if current_time - last_time >= fps_update_interval:
                    elapsed = current_time - start_time
                    avg_fps = frame_count / elapsed
                    current_fps = 1.0 / (time.time() - frame_start_time)
                    print(f"\rProcessing: Avg FPS: {avg_fps:.1f}, Current FPS: {current_fps:.1f}", end="")
                    last_time = current_time
                cv2.imshow('Basketball Shot Tracking', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            frame_count += 1
        total_time = time.time() - start_time
        print(f"\nProcessing complete:")
        print(f"Total frames: {frame_count}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average FPS: {frame_count / total_time:.1f}")
        print(f"Original video FPS: {self.video_fps}")
        print(f"Processing speed: {((frame_count / total_time) / self.video_fps):.1f}x realtime")
        self.cap.release()
        cv2.destroyAllWindows()
        self.print_stats()

    def print_stats(self):
        print(f"\nFinal Statistics:")
        print(f"Total Shots: {self.total_shots}")
        print(f"Made Shots: {self.made_shots}")
        print(f"Missed Shots: {self.missed_shots}")
        if self.total_shots > 0:
            percentage = (self.made_shots / self.total_shots) * 100
        else:
            percentage = 0
        print(f"Shooting Percentage: {percentage:.1f}%")

if __name__ == '__main__':
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
