import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import matplotlib.pyplot as plt
from rtmlib import Body

# Constants and configurations
openpose_skeleton = True  # True for openpose-style, False for mmpose-style
device = 'mps'  # cpu, cuda, mps
backend = 'onnxruntime'  # opencv, onnxruntime, openvino
ankle_indices = [10, 13]
body = Body(to_openpose=openpose_skeleton, mode='balanced', backend=backend, device=device)
g = 9.81  # m/s²


def free_fall(t, h0):
    return 0.5 * g * t ** 2 + h0


def cal_height(t):
    return (0.5 * g * (t / 2) ** 2) * 100


def draw_points(img, keypoints, scores):
    for idx in ankle_indices:
        if len(keypoints[0]) > idx and scores[0][idx] > 0.5:
            x, y = int(keypoints[0][idx][0]), int(keypoints[0][idx][1])
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)  # Draw a red dot to mark the ankle joint


def video_parser(video_path: str):
    ankle_y_coords = []  # coordinates of the ankle
    frame_times = []  # frame time

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    time_elapsed = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        keypoints, scores = body(frame)
        img_show = frame.copy()

        # Only draw ankle joint
        draw_points(img_show, keypoints, scores)

        # Obtain the position of the ankle joint
        left_ankle = keypoints[0][10] if len(keypoints[0]) > 10 else None
        right_ankle = keypoints[0][13] if len(keypoints[0]) > 13 else None

        if left_ankle is not None and right_ankle is not None:
            # Take the average vertical axis of the left and right ankle joints
            ankle_y = (left_ankle[1] + right_ankle[1]) / 2
            # Reverse the Y coordinate
            ankle_y = frame.shape[0] - ankle_y
            ankle_y_coords.append(ankle_y)
            frame_times.append(time_elapsed)

        time_elapsed += 1 / fps

        cv2.imshow('Tracking', img_show)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return frame_times, ankle_y_coords


def clean_frame(frame_times, ankle_y_coords):
    # Data cleaning: identifying jump intervals
    y_diff = np.diff(ankle_y_coords)  # Calculate the change in adjacent points
    jump_start = np.where(y_diff > 15)[0][0]  # Find the first obvious point of increase
    jump_end = np.where(ankle_y_coords[jump_start:] < ankle_y_coords[jump_start])[0][0] + jump_start

    filtered_times = frame_times[jump_start:jump_end]
    filtered_heights = ankle_y_coords[jump_start:jump_end]

    # The height of the initial point
    initial_ankle_y = filtered_heights[0]
    # Relative height
    filtered_heights = [y - initial_ankle_y for y in filtered_heights]

    x_data = np.array(filtered_times) - filtered_times[0]
    y_data = np.array(filtered_heights)

    return x_data, y_data, filtered_times, filtered_heights


def fit_data(x_data, y_data, filtered_times):
    # Quadratic polynomial fitting
    coeffs = np.polyfit(x_data, y_data, 2)

    fit_times = np.linspace(filtered_times[0], filtered_times[-1], 100)
    fit_heights = np.polyval(coeffs, fit_times - filtered_times[0])

    return coeffs, fit_times, fit_heights


def show_results_gui(height, duration, max_velocity):
    # Create a simple GUI to display the results
    root = tk.Tk()
    root.title("Jump Analysis Results")
    root.geometry("1000x700")  # Increased window size for better visibility

    # Set the entire background color to dark orange
    dark_orange = "#F28500"  # Hex code for rgb(242, 133, 0)
    root.configure(bg=dark_orange)

    # Add a title label with a larger font and white text
    title_label = tk.Label(root, text="Jump Analysis Results", font=("Arial", 36, "bold"), bg=dark_orange, fg="white")
    title_label.pack(pady=40)

    # Add result labels with larger fonts and white text
    tk.Label(root, text=f"Jump Height: {height:.2f} cm", font=("Arial", 24), bg=dark_orange, fg="white").pack(pady=20)
    tk.Label(root, text=f"Flight Time: {duration:.2f} s", font=("Arial", 24), bg=dark_orange, fg="white").pack(pady=20)
    tk.Label(root, text=f"Max Velocity: {max_velocity:.2f} m/s", font=("Arial", 24), bg=dark_orange, fg="white").pack(pady=20)

    # Add a button with a larger font and white text
    tk.Button(root, text="OK", command=root.destroy, font=("Arial", 20, "bold"), bg="white", fg=dark_orange).pack(pady=50)

    root.mainloop()


if __name__ == '__main__':
    # test video
    video_path = 'vert.mp4'
    # parse key points
    frame_times, ankle_y_coords = video_parser(video_path)
    # Extract the start and end frames of the jump
    x_data, y_data, filtered_times, filtered_heights = clean_frame(frame_times, ankle_y_coords)
    # Fit the relationship between h and t based on a quadratic function
    coeffs, fit_times, fit_heights = fit_data(x_data, y_data, filtered_times)
    # Flight time
    duration = fit_times[-1] - fit_times[0]
    # Jumping height(cm)
    height = cal_height(duration)
    max_height = np.max(fit_heights)
    cm_per_pixel = height / max_height
    filtered_heights = [y * cm_per_pixel for y in filtered_heights]
    fit_heights = [y * cm_per_pixel for y in fit_heights]
    # Draw a comparison chart
    plt.figure(figsize=(10, 5))
    plt.scatter(filtered_times, filtered_heights, color='b', label='Cleaned Ankle Data')
    plt.plot(fit_times, fit_heights, color='r', linestyle='-', label=f'Polyfit: h = {coeffs[0]*cm_per_pixel:.2f} t^2 + {coeffs[1]*cm_per_pixel:.2f} t + {coeffs[2]*cm_per_pixel:.2f}')
    plt.xlabel('Time (s)')
    plt.ylabel('Relative Height(cm)')
    plt.title('Jump Motion Polynomial Fitting')
    plt.legend()
    plt.grid()
    plt.text(0.01, 0.95, f"Jump Height: {height:.2f} cm",
             fontsize=12, ha='left', va='top', color='red',
             transform=plt.gca().transAxes)

    plt.text(0.01, 0.90, f"Flight Time: {duration:.2f} s",
             fontsize=12, ha='left', va='top', color='green',
             transform=plt.gca().transAxes)

    plt.text(0.01, 0.85, f"Max velocity: {coeffs[1] * cm_per_pixel / 100:.2f} m/s",
             fontsize=12, ha='left', va='top', color='blue',
             transform=plt.gca().transAxes)
    print(
        f'\nJump Height: {height:.2f} cm',
        f'\nFlight Time: {duration:.2f} s',
        f'\nMax velocity: {coeffs[1]*cm_per_pixel/100:.2f} m/s'
    )
    plt.show()

    # Show results in GUI
    max_velocity = coeffs[1] * cm_per_pixel / 100
    show_results_gui(height, duration, max_velocity)
