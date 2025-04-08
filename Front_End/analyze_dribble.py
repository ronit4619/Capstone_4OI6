import cv2  # OpenCV library for video processing
from ultralytics import YOLO  # YOLO model for object detection
import numpy as np  # NumPy for numerical operations
import time  # For calculating FPS dynamically

from flask import Flask, Response
app = Flask(__name__)

class DribbleCounter:
    def __init__(self, save_video=False):
        # Load the YOLO model with a pre-trained weights file
        self.model = YOLO("dribble_counting.pt")
        
        # Open the video file for processing
        #self.cap = cv2.VideoCapture("ron2.mp4")
        self.cap = cv2.VideoCapture(0)

        
        # Set the input video FPS (frames per second)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        
        # Initialize variables for tracking ball position and dribble count
        self.prev_x_center = None  # Previous x-coordinate of the ball's center
        self.prev_y_center = None  # Previous y-coordinate of the ball's center
        self.prev_delta_y = None  # Previous change in y-coordinate
        self.dribble_count = -1  # Start the dribble counter at -1
        self.dribble_threshold = 0.1  # Threshold for detecting a dribble based on y-coordinate changes
        self.dribble_counted = False  # Flag to ensure a dribble is counted only once per motion cycle

        # Set a fixed FPS for the output video (e.g., 60 FPS)
        self.output_fps = 60

        # Get video properties for saving the output video
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Width of the video frames
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Height of the video frames

        # Save video flag
        self.save_video = save_video

        # Initialize the video writer only if saving is enabled
        if self.save_video:
            self.output_writer = cv2.VideoWriter(
                "ron_out2_2.avi",  # Output file name
                cv2.VideoWriter_fourcc(*"XVID"),  # Codec for the video format
                self.output_fps,  # Fixed FPS for the output video
                (self.frame_width, self.frame_height)  # Frame size
            )

        # Initialize a Kalman filter for tracking the ball's position
        self.kalman = cv2.KalmanFilter(4, 2)  # 4 dynamic parameters (x, y, dx, dy), 2 measured parameters (x, y)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)  # Measurement matrix
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)  # Transition matrix
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03  # Process noise covariance matrix

        # Add a counter to track consecutive frames without detection
        self.no_detection_frames = 0
        self.no_detection_threshold = 10  # Number of frames before interpolation kicks in

    def run(self):
        # Start time for calculating FPS
        start_time = time.time()
        frame_count = 0  # Counter for the number of processed frames

        # Process video frames until the video ends or the user quits
        while self.cap.isOpened():
            success, frame = self.cap.read()  # Read a frame from the video
            if success:
                frame_count += 1  # Increment the frame counter

                # Convert the frame to grayscale for processing
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Perform object detection using the YOLO model
                results_list = self.model(frame, verbose=False, conf=0.55)  # Confidence threshold set to 0.55
                ball_detected = False  # Flag to indicate if the ball is detected in the current frame

                if results_list:
                    # Iterate through the detection results
                    for results in results_list:
                        for bbox in results.boxes.xyxy:  # Bounding box coordinates
                            x1, y1, x2, y2 = bbox[:4]  # Extract bounding box coordinates
                            x_center = (x1 + x2) / 2  # Calculate the x-coordinate of the center
                            y_center = (y1 + y2) / 2  # Calculate the y-coordinate of the center

                            # Update the Kalman filter with the detected position
                            measurement = np.array([[np.float32(x_center)], [np.float32(y_center)]])
                            self.kalman.correct(measurement)
                            ball_detected = True  # Ball detected in this frame
                            self.no_detection_frames = 0  # Reset the no-detection counter

                            # Predict the next position using the Kalman filter
                            prediction = self.kalman.predict()
                            predicted_x, predicted_y = prediction[0, 0], prediction[1, 0]

                            # Print the predicted ball coordinates
                            print(f"Ball coordinates: (x={predicted_x:.2f}, y={predicted_y:.2f})")
                            
                            # Update the dribble count based on the predicted position
                            self.update_dribble_count(predicted_x, predicted_y)
                            self.prev_x_center = predicted_x
                            self.prev_y_center = predicted_y

                            # Draw the predicted position on the frame
                            cv2.circle(frame, (int(predicted_x), int(predicted_y)), 5, (0, 255, 0), -1)  # Green circle

                if not ball_detected:
                    self.no_detection_frames += 1  # Increment no-detection counter

                    # Only use interpolation if the ball is not detected for a prolonged period
                    if self.no_detection_frames >= self.no_detection_threshold:
                        prediction = self.kalman.predict()
                        predicted_x, predicted_y = prediction[0, 0], prediction[1, 0]
                        print(f"Ball prediction (no detection): (x={predicted_x:.2f}, y={predicted_y:.2f})")
                        
                        # Update the dribble count based on the predicted position
                        self.update_dribble_count(predicted_x, predicted_y)
                        self.prev_x_center = predicted_x
                        self.prev_y_center = predicted_y

                        # Draw the predicted position on the frame
                        cv2.circle(frame, (int(predicted_x), int(predicted_y)), 5, (0, 0, 255), -1)  # Red circle for prediction

                # Display the dribble count on the frame
                # Add a white background rectangle for better text readability
                text = f"Dribble Count: {self.dribble_count}"
                font = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 2
                thickness = 4
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x, text_y = 10, self.frame_height - 30  # Position of the text moved down
                box_coords = ((text_x - 10, text_y - text_size[1] - 10), (text_x + text_size[0] + 10, text_y + 10))
                cv2.rectangle(frame, box_coords[0], box_coords[1], (255, 255, 255), cv2.FILLED)  # White rectangle

                # Add the black text on top of the white background
                cv2.putText(
                    frame,
                    text,
                    (text_x, text_y),
                    font,
                    font_scale,
                    (0, 0, 0),  # Text color (black)
                    thickness,
                    cv2.LINE_AA,
                )

                # Write the annotated frame to the output video if saving is enabled
                if self.save_video:
                    self.output_writer.write(frame)

                # Display the frame in a window
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
            
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

                # Exit the loop if the user presses 'q'
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                print("Failed to read frame from webcam.")
                break

        # Calculate and print the actual FPS of the processing
        elapsed_time = time.time() - start_time
        actual_fps = frame_count / elapsed_time
        print(f"Actual FPS: {actual_fps:.2f}")

        # Release resources
        self.cap.release()
        if self.save_video:
            self.output_writer.release()  # Release the video writer
        cv2.destroyAllWindows()

    def update_dribble_count(self, x_center, y_center):
        # Update the dribble count based on the ball's vertical movement
        if self.prev_y_center is not None:
            delta_y = y_center - self.prev_y_center  # Change in y-coordinate

            # Check for a downward motion indicating a dribble
            if (
                self.prev_delta_y is not None
                and self.prev_delta_y > self.dribble_threshold  # Previous movement was upward
                and delta_y < -self.dribble_threshold  # Current movement is downward
                and not self.dribble_counted  # Ensure we haven't already counted this dribble
            ):
                self.dribble_count += 1  # Increment the dribble count
                self.dribble_counted = True  # Mark this dribble as counted

            # Reset the flag when the motion changes direction (upward movement starts)
            if delta_y > self.dribble_threshold:
                self.dribble_counted = False

            self.prev_delta_y = delta_y  # Update the previous change in y-coordinate


dribble_counter = DribbleCounter(save_video=False)

@app.route('/video')
def video_feed():
    return Response(dribble_counter.run(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("Starting dribble counter stream on http://localhost:8002/video")
    app.run(host="0.0.0.0", port=8002)
