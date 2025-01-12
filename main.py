import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton
from PyQt5.QtCore import Qt

# Import the main.py file's code
from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Stylish Start Button GUI')
        self.setGeometry(100, 100, 300, 200)

        # Create Start Button
        self.start_button = QPushButton('Start', self)
        self.start_button.setGeometry(50, 70, 100, 40)
        self.start_button.clicked.connect(self.run_main)

        # Create Exit Button
        self.exit_button = QPushButton('Exit', self)
        self.exit_button.setGeometry(160, 70, 100, 40)
        self.exit_button.clicked.connect(self.exit_program)

        # Style both buttons similarly
        button_style = """
            QPushButton {
                background-color: #007AFF;
                color: white;
                border-radius: 20px;
                font-size: 16px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #005BBB;
            }
            QPushButton:pressed {
                background-color: #004BA0;
            }
        """
        self.start_button.setStyleSheet(button_style)
        self.exit_button.setStyleSheet(button_style)

    def run_main(self):
        # Run your main function here
        video_frames = read_video('input_videos/video3.mp4')

        # Initialize Tracker
        tracker = Tracker('models/best_12_30.pt')

        tracks = tracker.get_object_tracks(video_frames,
                                           read_from_stub=True,
                                           stub_path='stubs/track_stubs.pkl')
        # Get object positions 
        tracker.add_position_to_tracks(tracks)

        # Camera movement estimator
        camera_movement_estimator = CameraMovementEstimator(video_frames[0])
        camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                  read_from_stub=True,
                                                                                  stub_path='stubs/camera_movement_stub.pkl')
        camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

        # View Transformer
        view_transformer = ViewTransformer()
        view_transformer.add_transformed_position_to_tracks(tracks)

        # Interpolate Ball Positions
        tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

        # Speed and distance estimator
        speed_and_distance_estimator = SpeedAndDistance_Estimator()
        speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

        # Assign Player Teams
        team_assigner = TeamAssigner()
        team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
        
        for frame_num, player_track in enumerate(tracks['players']):
            for player_id, track in player_track.items():
                team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
                tracks['players'][frame_num][player_id]['team'] = team
                tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

        # Assign Ball Acquisition
        player_assigner = PlayerBallAssigner()
        team_ball_control = []
        for frame_num, player_track in enumerate(tracks['players']):
            ball_bbox = tracks['ball'][frame_num][1]['bbox']
            assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

            if assigned_player != -1:
                tracks['players'][frame_num][assigned_player]['has_ball'] = True
                team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
            else:
                if team_ball_control:
                    team_ball_control.append(team_ball_control[-1])
                else:
                    team_ball_control.append(None)  # or any default value
        team_ball_control = np.array(team_ball_control)

        # Draw output 
        output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
        output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
        speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

        # Save video
        save_video(output_video_frames, 'output_videos/output_video.avi')

        print("Video processing completed!")
        
    def exit_program(self):
        # Close the application
        QApplication.quit()

# Entry point for the PyQt application
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
