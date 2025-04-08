from ultralytics import YOLO
import cv2
import math
import numpy as np
from collections import deque
import os

#ultralytics
#opencv-python
#numpy

#classes available in the model: ["ball", "made", "person", "rim", "shoot"]


# Helper functions moved from helper.py
def distance(p1, p2):
    """
    Args:
    p1 (x,y) and p2 (x,y)
    """
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def is_increasing_distances(point, points_array):
    """
    Args:
    point (tuple): (x,y)
    points_array (list): List of tuples [(x_1, y_1), (x_2, y_2), ...].

    Returns:
    bool: True if the distances are strictly increasing, False otherwise.
    """
    distances = [distance(point, (x_i, y_i)) for (x_i, y_i) in points_array]
    for i in range(1, len(distances)):
        if distances[i] <= distances[i - 1]:
            return False
    return True

def is_ball_above_rim(ball, rim):
    """
    Args: 
    ball (cx, cy, frame)
    rim (x1, y1, x2, y2, frame)
    """
    return ball[1] < rim[1]

def is_ball_below_rim(ball, rim):
    """
    Args: 
    ball (cx, cy, frame)
    rim (x1, y1, x2, y2, frame)
    """
    return ball[1] > rim[3]

def is_made_shot(above_rim, below_rim, rim):
    """
    Args:
    above_rim (cx, cy, frame)
    below_rim (cx, cy, frame)
    rim (x1, y1, x2, y2, frame)
    """
    x1, y1, x2 = rim[0], rim[1], rim[2]
    cx1, cy1, cx2, cy2 = above_rim[0], above_rim[1], below_rim[0], below_rim[1]

    m = (cy2-cy1)/(cx2-cx1)
    b = cy1 - m*cx1
    x = (y1 - b) / m

    return x1 < x and x < x2

def write_text_with_background(img, text, location, font_face, font_scale, text_color, background_color, thickness):
    (tw, th), baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
    cv2.rectangle(img, (location[0], location[1] - th - baseline), (location[0] + tw, location[1] + baseline), background_color, -1)
    cv2.putText(img, text, location, font_face, font_scale, text_color, thickness)

def get_available_filename(output_dir, base_name, extension):
    counter = 1
    output_path = os.path.join(output_dir, f"{base_name}.{extension}")
    while os.path.exists(output_path):
        output_path = os.path.join(output_dir, f"{base_name}{counter}.{extension}")
        counter += 1
    return output_path

# Load Video
video_path = 'input_vids/steph.mp4'
cap = cv2.VideoCapture(video_path)
#cap = cv2.VideoCapture(0)


# Stuff for output video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

output_path = get_available_filename('output_vids', 'output', 'avi')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

model = YOLO("shot_counter_vid.pt")

# "made" class doesn't work very well
classnames = ["ball", "made", "person", "rim", "shoot"]

total_attempts = 0
total_made = 0

frame = 0

# In the format [x_center, y_center, frame]
ball_position = deque(maxlen=30)
shoot_position = deque(maxlen=30)
# In the format [x1, y1, x2, y2, frame]
rim_position = deque(maxlen=30)

ball_above_rim = None

overlay = None

while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)
    detections = np.empty((0,5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            
            # Bounding Box and confidence
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1
            conf = math.ceil(box.conf[0] * 100) / 100

            # Class name
            cls = int(box.cls[0])
            current_class = classnames[cls]

            cx, cy = x1+w // 2, y1+h // 2

            # Detecting the "shoot" action
            if current_class == "shoot" and conf>0.4:
                shoot_position.append([cx, cy, frame])
            
            # Check if ball is detected
            if current_class == "ball" and conf>0.4:
                ball_position.append([cx, cy, frame])

                # Draw the center point
                cv2.circle(img, (cx, cy), 5, (0, 0, 200), cv2.FILLED)

            # Check if rim is detected
            if current_class == "rim" and conf>0.4:
                rim_position.append([x1, y1, x2, y2, frame])
            
            # Draw bounding boxes and classnames
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 4)  # Changed color to white and thickness to 4
            write_text_with_background(img, f'{current_class.upper()} {conf}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), (255, 255, 255), 1)


    # Checks if distance from shoot position and ball keeps increasing after shot attempt
    # Checks if last time "shoot" was detected was five frames ago
    if shoot_position and shoot_position[-1][2] == frame - 5:
        last_ball_pos = [(cx, cy) for cx, cy, frame in list(ball_position)[-5:]]
        if is_increasing_distances((shoot_position[-1][0], shoot_position[-1][1]), last_ball_pos):
            total_attempts += 1
            print(f"Shot attempt detected. Total attempts: {total_attempts}")

    # This means that ball was above rim (or between lower and higher rim bound) in last frame and is now below rim
    if ball_above_rim and ball_position and rim_position and is_ball_below_rim(ball_position[-1], rim_position[-1]):
        if is_made_shot(ball_above_rim, ball_position[-1], rim_position[-1]):
            total_made += 1
            print(f"Shot made! Total made: {total_made}")
        ball_above_rim = None

    # By doing it through an if statement instead of just assignment, the variable ball_above_rim remains true when
    # lower_rim_bound < ball < higher_rim_bound
    # Check if ball_position and rim_position are not empty before accessing
    if ball_position and rim_position and is_ball_above_rim(ball_position[-1], rim_position[-1]):
        ball_above_rim = ball_position[-1]
    
    # Update the position, font size, and background for the text
    write_text_with_background(img, f'Shots Attempted: {str(total_made)}', (frame_width - 250, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), (255, 255, 255), 1)
    write_text_with_background(img, f'Shots Made: {total_made}', (frame_width - 250, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), (255, 255, 255), 1)
    shots_missed = max(0, total_attempts - total_made)  # Ensure shots missed is not negative
    #write_text_with_background(img, f'Shots Missed: {shots_missed}', (frame_width - 250, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), (255, 255, 255), 2)

    # Adds circles on ball position every 5 frames
    if overlay is None:
        overlay = np.zeros_like(img, dtype=np.uint8)

    # Draws a path for the balls
    if frame % 5 == 0:
        # Clear the overlay (reset to transparent)
        overlay = np.zeros_like(img, dtype=np.uint8)
        
        for pos in ball_position:
            cx, cy, pos_frame = pos
            if pos_frame % 5 == 0:
                cv2.circle(overlay, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
    
    frame += 1

    # Blend the overlay onto the main frame
    blended_img = cv2.addWeighted(img, 1.0, overlay, 1, 0)

    cv2.imshow("Image", blended_img)

    # Write the frame to the video file
    out.write(blended_img)

    # To watch video frame by frame
    # cv2.waitKey(0)

    # To watch video continuosly
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
