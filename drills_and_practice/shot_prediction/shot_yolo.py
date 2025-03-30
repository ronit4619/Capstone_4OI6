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
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_width, frame_height))

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

        for x, (posX, posY) in enumerate(zip(posListX, posListY)):
            pos = (posX, posY)
            cv2.circle(img, tuple(pos), 10, (0, 255, 0), cv2.FILLED)
            if x == 0:
                cv2.line(img, tuple(pos), tuple(pos), (255, 0, 0), 5)
            else:
                cv2.line(img, tuple(pos), (posListX[x - 1], posListY[x - 1]), (255, 0, 0), 2)

        for x in xList:
            y = int(A * x ** 2 + B * x + C)
            cv2.circle(img, (x, y), 2, (0, 0, 255), cv2.FILLED)

        if len(posListX) > 1:
            # Check for upward motion indicating a shot
            if posListY[-1] < posListY[-2] - 10:  # Threshold to detect significant upward motion
                shot_detected = True
            else:
                shot_detected = False

        if shot_detected and len(posListX) < 10:
            # Predictions
            a = A
            b = B
            c = C - 590

            discriminant = b ** 2 - (4 * a * c)
            if discriminant >= 0:  # Check if the discriminant is non-negative
                x = int((-b - math.sqrt(discriminant)) / (2 * a))
                prediction = 300 < x < 400
            elif discriminant <= 0:  # Check if the discriminant is non-negative
                x = int(((-b - math.sqrt(0-discriminant)) / (2 * a)))
                prediction = 300 < x < 400
            else:
                prediction = False  # If discriminant is negative, no valid prediction

        if prediction:
            cvzone.putTextRect(img, "Basket", (50, 150), scale=5, thickness=5, colorR=(0, 200, 0), offset=20)
        else:
            cvzone.putTextRect(img, "No Basket", (50, 150), scale=5, thickness=5, colorR=(0, 0, 200), offset=20)

    # Display
    img = cv2.resize(img, (0, 0), None, 0.7, 0.7)
    cv2.imshow("Image", img)

    # Write the frame to the video file
    out.write(cv2.resize(img, (frame_width, frame_height)))

    cv2.waitKey(100)

# Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()