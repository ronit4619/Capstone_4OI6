from ultralytics import YOLO
import os
import cv2

# Load model
pt = os.path.join(os.getcwd(), "model_pt/shot_detection_v2.pt")
model = YOLO(pt)

# Inference
source = os.path.join(os.getcwd(), "testing-datasets/alan.mp4")
results = model(source, save=True, conf=0.3, show_labels=True, boxes=True, stream=False)

# Open the video file
video_path = os.path.join(os.getcwd(), "testing-datasets/alan.mp4")
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
#
#appnope==0.1.3
#asttokens==2.2.1
#backcall==0.2.0
#Brotli==1.1.0
#certifi==2022.12.7
#chardet==4.0.0
#charset-normalizer==3.2.0
#comm==0.1.4
#contourpy==1.1.0
#cycler==0.10.0
#debugpy==1.6.7.post1
#decorator==4.4.2
#executing==1.2.0
#ffmpeg-python==0.2.0
#filelock==3.12.2
#fonttools==4.42.0
#future==0.18.3
#idna==2.10
#imageio==2.31.3
#imageio-ffmpeg==0.4.8
#ipykernel==6.25.1
#ipython==8.14.0
#jedi==0.19.0
#Jinja2==3.1.2
#jupyter_client==8.3.0
#jupyter_core==5.3.1
#kiwisolver==1.4.4
#MarkupSafe==2.1.3
#matplotlib==3.7.2
#matplotlib-inline==0.1.6
#mpmath==1.3.0
#mutagen==1.47.0
#ndi-python==5.1.1.5
#nest-asyncio==1.5.7
#networkx==3.1
#numpy==1.25.2
#onnx==1.14.0
#opencv-python==4.8.0.76
#opencv-python-headless==4.8.0.74
#packaging==23.1
#pandas==2.0.3
#parso==0.8.3
#pexpect==4.8.0
#pickleshare==0.7.5
#Pillow==9.5.0
#platformdirs==3.10.0
#proglog==0.1.10
#prompt-toolkit==3.0.39
#protobuf==4.24.1
#psutil==5.9.5
#ptyprocess==0.7.0
#pure-eval==0.2.2
#py-cpuinfo==9.0.0
#pycryptodomex==3.18.0
#Pygments==2.16.1
#pyparsing==2.4.7
#python-dateutil==2.8.2
#python-dotenv==1.0.0
#pytz==2023.3
#PyYAML==6.0.1
#pyzmq==25.1.1
#PyQt5==5.15.10
#PyQt5-Qt5==5.15.11
#PyQt5-sip==12.13.0
#requests==2.31.0
#requests-toolbelt==1.0.0
#roboflow==1.1.6
#scipy==1.11.1
#seaborn==0.12.2
#six==1.16.0
#stack-data==0.6.2
#supervision==0.14.0
#sympy==1.12
#torch==2.0.1
#torchvision==0.15.2
#tornado==6.3.3
#tqdm==4.66.1
#traitlets==5.9.0
#typing_extensions==4.7.1
#tzdata==2023.3
#ultralytics==8.0.156
#urllib3==2.0.4
#wcwidth==0.2.6
#websockets==11.0.3
