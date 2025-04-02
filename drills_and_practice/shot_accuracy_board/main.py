import cv2
from ultralytics import YOLO
model = YOLO('yolov5n.pt')  # You can use yolov8s.pt or yolov8m.pt for better accuracy
model1=YOLO('ball_rimV8.pt')  # Load your custom model

# Constants for line positions (x coordinates)
THREE_POINT_X = 685  # Adjust based on video
FREE_THROW_X = 830   # Adjust as needed

# Constants
PIXELS_PER_METER = 150  # Adjust after calibration
FREE_THROW_DISTANCE = 6.32  # in meters
THREE_POINT_DISTANCE = 7.24  # in meters

# Line colors (BGR) and thickness
THREE_POINT_COLOR = (255, 191, 0)  # Cyan-like
FREE_THROW_COLOR = (0, 0, 255)     # Red
THICKNESS = 6

# Optional: label font
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Desired output dimensions (720p)
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

# Load video
cap = cv2.VideoCapture('shot.mp4')  # Replace with actual path

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # Run YOLOv8 inference (on BGR image)
    results = model(frame)[0]#Get first result from batch
    results1 = model1(frame)[0]#Get first result from batch
    # Loop over detections and draw if class is 'person' (class 0)
    for box in results.boxes:
        cls = int(box.cls[0])
        if cls == 0:  # 'person'
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, 'Person', (x1, y1 - 10), FONT, 0.8, (0, 255, 0), 2)

    best_rim = None
    highest_confidence = 0.0

    for box in results1.boxes:
        cls = int(box.cls[0])
        confidence = float(box.conf[0])  # Get the confidence score
        if cls == 1 and confidence >= 0.55:  # 'rim' with confidence >= 55%
            if confidence > highest_confidence:
                highest_confidence = confidence
                best_rim = box

    if best_rim:
        x1, y1, x2, y2 = map(int, best_rim.xyxy[0])  # Calculate rim coordinates here
        rim_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        last_rim_center = rim_center  # Update the last detected rim center
        cv2.circle(frame, rim_center, 10, (0, 140, 255), -1)
        cv2.putText(frame, 'Rim', (rim_center[0] + 10, rim_center[1]), FONT, 0.8, (0, 140, 255), 2)

    # Get all 'person' boxes
    person_boxes = [box for box in results.boxes if int(box.cls[0]) == 0]
    
    if rim_center:
        rim_x = rim_center[0]

        # Calculate line positions in pixels
        free_throw_x = rim_x - int(FREE_THROW_DISTANCE*PIXELS_PER_METER)
        three_point_x = rim_x - int(THREE_POINT_DISTANCE*PIXELS_PER_METER)

        # Draw free throw and 3PT lines
        cv2.line(frame, (three_point_x, 0), (three_point_x, height), (255, 191, 0), 6)
        cv2.putText(frame, '3PT Line', (three_point_x + 10, 50), FONT, 1, (255, 191, 0), 2)

        cv2.line(frame, (free_throw_x, 0), (free_throw_x, height), (0, 0, 255), 6)
        cv2.putText(frame, 'FT Line', (free_throw_x + 10, 100), FONT, 1, (0, 0, 255), 2)

    
        if person_boxes:
            # Pick the closest person (tallest bbox)
            shooter_box = max(person_boxes, key=lambda b: b.xyxy[0][3] - b.xyxy[0][1])
            x1, y1, x2, y2 = map(int, shooter_box.xyxy[0])
            cx = int((x1 + x2) / 2)
        
            # Draw the shooter's bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.circle(frame, (cx, int((y1 + y2)/2)), 6, (255, 255, 255), -1)
        
            # Classify shot type based on X-position (right-side hoop assumption)
            if abs(cx - FREE_THROW_X) < 50:
                shot_type = "Free Throw"
                color = (0, 0, 255)
            elif free_throw_x < cx <= three_point_x:
                shot_type = "2 Pointer"
                color = (255, 255, 0)
            elif cx <= three_point_x:
                shot_type = "Three Point"
                color = (255, 191, 0)
            else:
                shot_type = "Unknown"
                color = (128, 128, 128)

    
        # Label the shot type
        cv2.putText(frame, f'{shot_type}', (x1, y2 + 30), FONT, 1.0, color, 3)

    # Resize to 720p
    frame_resized = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

    # Display frame
    cv2.imshow('Court Lines + YOLO Person Detection (720p)', frame_resized)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

