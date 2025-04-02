import cv2
import numpy as np
from ultralytics import YOLO

def on_trackbar_change(_):
    pass

def get_arc_points(center, axes, angle_deg, start_deg, end_deg, num_points=100):
    pts = []
    angle_rad = np.deg2rad(angle_deg)
    for t in np.linspace(np.deg2rad(start_deg), np.deg2rad(end_deg), num_points):
        x = axes[0] * np.cos(t)
        y = axes[1] * np.sin(t)
        xr = x * np.cos(angle_rad) - y * np.sin(angle_rad)
        yr = x * np.sin(angle_rad) + y * np.cos(angle_rad)
        pts.append((center[0] + xr, center[1] + yr))
    return np.array(pts)

# === CONFIG ===
video_path = "shot.mp4"  # Leave "" for webcam
video_mode = video_path != ""

# === SETUP ===
cap = cv2.VideoCapture(video_path if video_mode else 0)
if not cap.isOpened():
    raise RuntimeError("Could not open video source.")

if not video_mode:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

composite_width, composite_height = 1920, 1080
control_width = 400
video_width = composite_width - control_width
video_height = composite_height

cv2.namedWindow('Overlay', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Overlay', composite_width, composite_height)

# === TRACKBARS ===
cv2.createTrackbar('FreeThrow X', 'Overlay', video_width // 2, video_width, on_trackbar_change)
cv2.createTrackbar('FreeThrow Y', 'Overlay', video_height // 4, video_height, on_trackbar_change)
cv2.createTrackbar('FreeThrow Major', 'Overlay', 80, video_width // 2, on_trackbar_change)
cv2.createTrackbar('FreeThrow Minor', 'Overlay', 40, video_height // 2, on_trackbar_change)
cv2.createTrackbar('FreeThrow Angle', 'Overlay', 0, 360, on_trackbar_change)

cv2.createTrackbar('ThreePoint X', 'Overlay', video_width // 2, video_width, on_trackbar_change)
cv2.createTrackbar('ThreePoint Y', 'Overlay', video_height // 2, video_height, on_trackbar_change)
cv2.createTrackbar('ThreePoint Major', 'Overlay', 160, video_width // 2, on_trackbar_change)
cv2.createTrackbar('ThreePoint Minor', 'Overlay', 80, video_height // 2, on_trackbar_change)
cv2.createTrackbar('ThreePoint Angle', 'Overlay', 0, 360, on_trackbar_change)

detection_enabled = False
processing = not video_mode
threshold = 15

# Load YOLO model
model = YOLO("best.pt")

# === ADJUST FIRST FRAME IF VIDEO ===
if video_mode and not processing:
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Unable to read first frame from video.")
    first_frame = cv2.resize(first_frame, (video_width, video_height))

    while True:
        display_frame = first_frame.copy()

        # Get trackbar values
        ft_x = cv2.getTrackbarPos('FreeThrow X', 'Overlay')
        ft_y = cv2.getTrackbarPos('FreeThrow Y', 'Overlay')
        ft_major = cv2.getTrackbarPos('FreeThrow Major', 'Overlay')
        ft_minor = cv2.getTrackbarPos('FreeThrow Minor', 'Overlay')
        ft_angle = cv2.getTrackbarPos('FreeThrow Angle', 'Overlay')
        tp_x = cv2.getTrackbarPos('ThreePoint X', 'Overlay')
        tp_y = cv2.getTrackbarPos('ThreePoint Y', 'Overlay')
        tp_major = cv2.getTrackbarPos('ThreePoint Major', 'Overlay')
        tp_minor = cv2.getTrackbarPos('ThreePoint Minor', 'Overlay')
        tp_angle = cv2.getTrackbarPos('ThreePoint Angle', 'Overlay')

        # Draw arcs
        cv2.ellipse(display_frame, (ft_x, ft_y), (ft_major, ft_minor), ft_angle, 0, 180, (0, 0, 255), 2)
        cv2.ellipse(display_frame, (tp_x, tp_y), (tp_major, tp_minor), tp_angle, 0, 180, (255, 0, 0), 2)

        # Build composite image
        composite = np.zeros((composite_height, composite_width, 3), dtype=np.uint8)
        composite[:, :control_width] = (50, 50, 50)
        composite[:, control_width:] = display_frame
        cv2.putText(composite, "Adjust curves, press 'p' to process, 'q' to quit",
                    (10, composite_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow('Overlay', composite)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):
            processing = True
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# === OUTPUT VIDEO SETUP ===
if video_mode:
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("output.avi", fourcc, fps, (composite_width, composite_height))
else:
    out = None

# === MAIN LOOP ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    video_frame = cv2.resize(frame, (video_width, video_height))

    # Get trackbar values
    ft_x = cv2.getTrackbarPos('FreeThrow X', 'Overlay')
    ft_y = cv2.getTrackbarPos('FreeThrow Y', 'Overlay')
    ft_major = cv2.getTrackbarPos('FreeThrow Major', 'Overlay')
    ft_minor = cv2.getTrackbarPos('FreeThrow Minor', 'Overlay')
    ft_angle = cv2.getTrackbarPos('FreeThrow Angle', 'Overlay')

    tp_x = cv2.getTrackbarPos('ThreePoint X', 'Overlay')
    tp_y = cv2.getTrackbarPos('ThreePoint Y', 'Overlay')
    tp_major = cv2.getTrackbarPos('ThreePoint Major', 'Overlay')
    tp_minor = cv2.getTrackbarPos('ThreePoint Minor', 'Overlay')
    tp_angle = cv2.getTrackbarPos('ThreePoint Angle', 'Overlay')

    # Draw arcs
    cv2.ellipse(video_frame, (ft_x, ft_y), (ft_major, ft_minor), ft_angle, 0, 180, (0, 0, 255), 2)
    cv2.ellipse(video_frame, (tp_x, tp_y), (tp_major, tp_minor), tp_angle, 0, 180, (255, 0, 0), 2)

    if detection_enabled:
        ft_points = get_arc_points((ft_x, ft_y), (ft_major, ft_minor), ft_angle, 0, 180)
        tp_points = get_arc_points((tp_x, tp_y), (tp_major, tp_minor), tp_angle, 0, 180)
        results = model(video_frame)[0]

        for box in results.boxes:
            cls_id = int(box.cls.cpu().numpy()[0])
            if cls_id != 2:  # Assuming class ID 2 corresponds to a person
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cv2.rectangle(video_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            bottom_center = (int((x1 + x2) / 2), int(y2))
            cv2.circle(video_frame, bottom_center, 5, (0, 255, 0), -1)

            # Calculate distances to the arcs
            ft_dist = np.min(np.sqrt((ft_points[:, 0] - bottom_center[0])**2 +
                                     (ft_points[:, 1] - bottom_center[1])**2))
            tp_dist = np.min(np.sqrt((tp_points[:, 0] - bottom_center[0])**2 +
                                     (tp_points[:, 1] - bottom_center[1])**2))

            # Determine if the person is on a line
            text = ""
            if ft_dist < threshold:
                text += "On Free Throw Line"
            if tp_dist < threshold:
                if text:
                    text += " & "
                text += "On Three Point Line"

            # Display the text if the person is on a line
            if text:
                cv2.putText(video_frame, text, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Build and show composite
    composite = np.zeros((composite_height, composite_width, 3), dtype=np.uint8)
    composite[:, :control_width] = (50, 50, 50)
    composite[:, control_width:] = video_frame

    cv2.imshow('Overlay', composite)
    if out:
        out.write(composite)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        detection_enabled = not detection_enabled
    elif key == ord('q'):
        break

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
