import cv2
import cvzone
from ultralytics import YOLO
import numpy as np
import math

# Initialize the Videos
cap = cv2.VideoCapture("Videos/made5.mp4")

# Load the YOLO model
model = YOLO("best.pt")

# Variables
posListX, posListY = [], []
xList = [item for item in range(0, 1300)]
prediction = False
shot_detected = False

# Get the width and height of the video frames
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, (frame_width, frame_height))

while True:
    # Grab the image
    success, img = cap.read()
    if not success:
        break

    # Use YOLO to detect the ball
    results = model(img)
    for result in results:
        for box in result.boxes:
            if box.cls == 0:  # Assuming class 0 is the ball
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                posListX.append(cx)
                posListY.append(cy)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

    if posListX:
        # Polynomial Regression
        A, B, C = np.polyfit(posListX, posListY, 2)

        # Draw trajectory as small neon orange dots
        for x, y in zip(posListX, posListY):
            cv2.circle(img, (x, y), 3, (0, 165, 255), cv2.FILLED)  # Neon orange (BGR: 0, 165, 255)

        # Draw predicted trajectory as neon blue dashes
        for i, x in enumerate(xList):
            if i % 5 == 0:  # Skip some points to create a dashed effect
                y = int(A * x ** 2 + B * x + C)
                cv2.circle(img, (x, y), 2, (255, 0, 0), cv2.FILLED)  # Neon blue (BGR: 255, 0, 0)

        if len(posListX) > 1:
            # Check for upward motion indicating a shot
            if posListY[-1] < posListY[-2] - 5:  # Reduced threshold for better sensitivity
                shot_detected = True
            else:
                shot_detected = False

        if shot_detected and len(posListX) < 15:  # Increased point limit for better prediction
            # Predictions
            a = A
            b = B
            c = C - 590

            discriminant = b ** 2 - (4 * a * c)
            #cv2.putText(img, f"Discriminant: {discriminant:.2f}", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            if discriminant >= 0:  # Check if the discriminant is non-negative
                x = int((-b + math.sqrt(discriminant)) / (2 * a))
                #cv2.putText(img, f"x: {x:.2f}", (50, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Expanded range for leniency near the rim
                prediction = 850 < x# Adjusted range for more leniency
            else:
                prediction = False  # If discriminant is negative, no valid prediction

        if prediction:
            cvzone.putTextRect(img, "Basket", (20, 100), scale=3, thickness=3, colorR=(255, 255, 255), colorT=(0, 0, 0), offset=10)
        else:
            cvzone.putTextRect(img, "No Basket", (20, 100), scale=3, thickness=3, colorR=(255, 255, 255), colorT=(0, 0, 0), offset=10)

    # Display
    # Draw lines for prediction range
    #cv2.line(img, (950, 0), (950, img.shape[0]), (255, 255, 0), 2)  # Left boundary
    #cv2.line(img, (1095, 0), (1095, img.shape[0]), (255, 255, 0), 2)  # Right boundary

    img = cv2.resize(img, (0, 0), None, 0.7, 0.7)
    cv2.imshow("Image", img)

    # Write the frame to the video file
    out.write(cv2.resize(img, (frame_width, frame_height)))

    cv2.waitKey(100)

# Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()