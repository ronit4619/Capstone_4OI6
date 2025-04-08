import cv2  # OpenCV library for video processing
from ultralytics import YOLO  # YOLO model for object detection
import numpy as np  # NumPy for numerical operations
import time  # For calculating FPS dynamically

class DribbleCounter:
    def __init__(self, save_video=False):
        # Load the YOLO model with a pre-trained weights file
        self.model = YOLO("dribble_counting.pt")
        
        # Open the video file for processing
        self.cap = cv2.VideoCapture("input.mp4")
        
        # Set the input video FPS (frames per second)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        
        # Initialize variables for tracking ball position and dribble count
        self.prev_x_center = None
        self.prev_y_center = None
        self.prev_delta_y = None
        self.dribble_count = 0
        self.dribble_threshold = 0.1
        self.dribble_counted = False

        # Set a fixed FPS for the output video (e.g., 60 FPS)
        self.output_fps = 60

        # Get video properties for saving the output video
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Save video flag
        self.save_video = save_video

        # Initialize the video writer only if saving is enabled
        if self.save_video:
            self.output_writer = cv2.VideoWriter(
                "test_out21.avi",  # Output file name
                cv2.VideoWriter_fourcc(*"XVID"),  # Codec for the video format
                self.output_fps,  # Fixed FPS for the output video
                (self.frame_width, self.frame_height)  # Frame size
            )

    def run(self):
        # Start time for calculating FPS
        start_time = time.time()
        frame_count = 0

        # Process video frames until the video ends or the user quits
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if success:
                frame_count += 1

                # Perform object detection using the YOLO model
                results_list = self.model(frame, verbose=False, conf=0.55)
                ball_detected = False

                if results_list:
                    for results in results_list:
                        for bbox in results.boxes.xyxy:
                            x1, y1, x2, y2 = bbox[:4]
                            x_center = (x1 + x2) / 2
                            y_center = (y1 + y2) / 2

                            ball_detected = True

                            # Update the dribble count based on the detected position
                            self.update_dribble_count(x_center, y_center)
                            self.prev_x_center = x_center
                            self.prev_y_center = y_center

                            # Draw the detected position on the frame
                            cv2.circle(frame, (int(x_center), int(y_center)), 5, (0, 255, 0), -1)

                # Display the dribble count on the frame
                text = f"Dribble Count: {self.dribble_count}"
                font = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 2
                thickness = 4
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x, text_y = 10, self.frame_height - 30
                box_coords = ((text_x - 10, text_y - text_size[1] - 10), (text_x + text_size[0] + 10, text_y + 10))
                cv2.rectangle(frame, box_coords[0], box_coords[1], (255, 255, 255), cv2.FILLED)
                cv2.putText(
                    frame,
                    text,
                    (text_x, text_y),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness,
                    cv2.LINE_AA,
                )

                # Write the annotated frame to the output video if saving is enabled
                if self.save_video:
                    self.output_writer.write(frame)

                # Display the frame in a window
                cv2.imshow("YOLOv8 Inference", frame)

                # Exit the loop if the user presses 'q'
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                print("Failed to read frame from video.")
                break

        # Calculate and print the actual FPS of the processing
        elapsed_time = time.time() - start_time
        actual_fps = frame_count / elapsed_time
        print(f"Actual FPS: {actual_fps:.2f}")

        # Release resources
        self.cap.release()
        if self.save_video:
            self.output_writer.release()
        cv2.destroyAllWindows()

    def update_dribble_count(self, x_center, y_center):
        # Update the dribble count based on the ball's vertical movement
        if self.prev_y_center is not None:
            delta_y = y_center - self.prev_y_center

            if (
                self.prev_delta_y is not None
                and self.prev_delta_y > self.dribble_threshold
                and delta_y < -self.dribble_threshold
                and not self.dribble_counted
            ):
                self.dribble_count += 1
                self.dribble_counted = True

            if delta_y > self.dribble_threshold:
                self.dribble_counted = False

            self.prev_delta_y = delta_y

if __name__ == "__main__":
    save_option = input("Would you like to save the output video? (yes/no): ").strip().lower()
    save_video = save_option == "yes"

    dribble_counter = DribbleCounter(save_video=save_video)
    dribble_counter.run()