# Basketball Shot Tracking System

An AI-powered basketball shot tracking system that uses computer vision and pose estimation to detect and analyze basketball shots in real-time or from video files.

## Demo Output (ran on test.mp4)
https://drive.google.com/file/d/1-BCyAcxO6KmdRDS_rWXP1qdSK5gJhK8W/view?usp=drive_link

## Features

- Real-time and video file processing modes
- Multi-model detection system:
  - Player detection using YOLOv8
  - Basketball detection using custom trained model
  - Pose estimation for shot detection
- Advanced shot tracking:
  - Ball trajectory analysis
  - Pose-based shot detection
  - Adaptive FPS-based parameters
- Real-time statistics:
  - Total shots
  - Made shots
  - Missed shots
  - Shooting percentage

## Requirements

- Python 3.x
- Required packages (install via `requirements.txt`):
  ```
  opencv-python
  numpy
  ffmpeg-python
  torch
  torchvision
  torchaudio
  torchmetrics
  ultralytics
  scikit-learn
  mediapipe
  pandas
  matplotlib
  seaborn
  ```
- YOLO models:
  - `yolov8n.pt` - For player detection
  - `best.pt` - Custom trained model for basketball/rim detection
  - `yolo11n-pose.pt` - For pose estimation

## Installation

1. Clone the repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Download required YOLO models and place them in the project root (already in github hopefully):
   - `yolov8n.pt`
   - `best.pt`
   - `yolo11n-pose.pt`

## Usage

The system supports two modes of operation:

### Real-time Mode
```bash
python main.py --mode realtime --input INPUT_VIDEO
```
- Processes live video feed from camera
- Optional `--input` parameter to specify camera source

### Video Processing Mode
```bash
python main.py --mode video --input INPUT_VIDEO --output OUTPUT_VIDEO
```
- Required parameters:
  - `--input`: Path to input video file
  - `--output`: Path to save processed video

## Technical Details

The system uses a multi-stage detection and tracking approach:

1. **Frame Processing**:
   - Adaptive parameters based on video FPS
   - Configurable shot detection timeouts
   - Queue-based frame processing with threading support

2. **Shot Detection**:
   - Combines pose analysis and ball trajectory
   - Uses upward motion detection with frame-based smoothing
   - Tracks ball through entry and make detection boxes
   - Configurable parameters for shot verification

3. **Statistics Tracking**:
   - Real-time shot counting
   - Made/missed shot detection
   - Shooting percentage calculation

## Performance Metrics

The system tracks:
- Total shots attempted
- Shots made
- Shots missed
- Shooting percentage

## Limitations

- Requires proper lighting conditions
- Performance depends on:
  - Camera angle and position
  - Video quality
  - Processing hardware capabilities
- Shot detection accuracy may vary with complex movements or occlusions

## Contributing

Feel free to open issues or submit pull requests for improvements.