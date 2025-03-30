import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np
import math

# Initialize the Videos
cap = cv2.VideoCapture("Videos/vid (1).mp4")

# Create the color finder object
colorFinder = ColorFinder(False)
hsvVals = {"hmin": 12, "smin": 150, "vmin": 0, "hmax": 13, "smax": 255, "vmax": 255}

# Variables
posListX, posListY = [], []
xList = [item for item in range(0, 1300)]
prediction = False

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

    # img = cv2.imread("Ball.png")

    # img = img[0:900, :]

    # Find the color of the ball
    imgColor, mask = colorFinder.update(img, hsvVals)

    # Find location of the ball
    imgContours, contours = cvzone.findContours(img, mask, minArea=500)

    if contours:
        posListX.append(contours[0]["center"][0])
        posListY.append(contours[0]["center"][1])
        # print(cx, cy)

    if posListX:
        # !Polynomial Regression
        # * y = Ax^2 + Bx + C

        # Find the Coefficients
        A, B, C = np.polyfit(
            posListX, posListY, 2
        )  # give List and degree of Polynomial

        for x, (posX, posY) in enumerate(zip(posListX, posListY)):
            pos = (posX, posY)
            cv2.circle(imgContours, tuple(pos), 10, (0, 255, 0), cv2.FILLED)
            if x == 0:
                cv2.line(imgContours, tuple(pos), tuple(pos), (255, 0, 0), 5)
            else:
                cv2.line(
                    imgContours,
                    tuple(pos),
                    (posListX[x - 1], posListY[x - 1]),
                    (255, 0, 0),
                    2,
                )

        for x in xList:
            y = int(A * x ** 2 + B * x + C)
            cv2.circle(imgContours, (x, y), 2, (0, 0, 255), cv2.FILLED)

        if len(posListX) < 10:

            # Predictions
            #! X values 330 to 400  Y 590
            a = A
            b = B
            c = C - 590

            x = int((-b - math.sqrt(b ** 2 - (4 * a * c))) / (2 * a))
            prediction = 300 < x < 400

        if prediction:
            cvzone.putTextRect(
                imgContours,
                "Basket",
                (50, 150),
                scale=5,
                thickness=5,
                colorR=(0, 200, 0),
                offset=20,
            )
        else:
            cvzone.putTextRect(
                imgContours,
                "No Basket",
                (50, 150),
                scale=5,
                thickness=5,
                colorR=(0, 0, 200),
                offset=20,
            )

    # Display
    imgContours = cv2.resize(imgContours, (0, 0), None, 0.7, 0.7)
    # mask = cv2.resize(mask, (0, 0), None, 0.7, 0.7)

    # cv2.imshow("Image", img)
    cv2.imshow("ImageColor", imgContours)

    # Write the frame to the video file
    out.write(cv2.resize(imgContours, (frame_width, frame_height)))

    cv2.waitKey(100)

# Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()