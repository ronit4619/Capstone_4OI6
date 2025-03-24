from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
import torch

#cvzone==1.5.6
#ultralytics==8.0.26
#hydra-core>=1.2.0
#matplotlib>=3.2.2
#numpy>=1.18.5
#opencv-python==4.5.4.60
#Pillow>=7.1.2
#PyYAML>=5.3.1
#requests>=2.23.0
#scipy>=1.4.1
#torch>=1.7.0
#torchvision>=0.8.1
#tqdm>=4.64.0
#filterpy==1.4.5
#scikit-image==0.19.3
#lap==0.4.0

def get_device():
    """Select device: cuda, mps, or cpu"""
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    return device

def score(ball_pos, hoop_pos):
    x = []
    y = []
    rim_height = hoop_pos[-1][0][1] - 0.5 * hoop_pos[-1][3]

    #Get points above and below rim
    for i in reversed(range(len(ball_pos))):
        if ball_pos[i][0][1] < rim_height:
            x.append(ball_pos[i][0][0])
            y.append(ball_pos[i][0][1])
            if i + 1 < len(ball_pos):
                x.append(ball_pos[i + 1][0][0])
                y.append(ball_pos[i + 1][0][1])
            break

    #Create line from points
    if len(x) > 1:
        m, b = np.polyfit(x, y, 1)
        predicted_x = ((hoop_pos[-1][0][1] - 0.5 * hoop_pos[-1][3]) - b) / m
        rim_x1 = hoop_pos[-1][0][0] - 0.4 * hoop_pos[-1][2]
        rim_x2 = hoop_pos[-1][0][0] + 0.4 * hoop_pos[-1][2]

        #Check if path crosses rim area
        if rim_x1 < predicted_x < rim_x2:
            return True
        #Check if ball enters rebound zone
        hoop_rebound_zone = 10
        if rim_x1 - hoop_rebound_zone < predicted_x < rim_x2 + hoop_rebound_zone:
            return True

    return False

#Detect if ball is below net
def detect_down(ball_pos, hoop_pos):
    y = hoop_pos[-1][0][1] + 0.5 * hoop_pos[-1][3]
    if ball_pos[-1][0][1] > y:
        return True
    return False

#Detect if ball is near backboard
def detect_up(ball_pos, hoop_pos):
    x1 = hoop_pos[-1][0][0] - 4 * hoop_pos[-1][2]
    x2 = hoop_pos[-1][0][0] + 4 * hoop_pos[-1][2]
    y1 = hoop_pos[-1][0][1] - 2 * hoop_pos[-1][3]
    y2 = hoop_pos[-1][0][1]

    if x1 < ball_pos[-1][0][0] < x2 and y1 < ball_pos[-1][0][1] < y2 - 0.5 * hoop_pos[-1][3]:
        return True
    return False

#Check if center point is near hoop
def in_hoop_region(center, hoop_pos):
    if len(hoop_pos) < 1:
        return False
    x = center[0]
    y = center[1]

    x1 = hoop_pos[-1][0][0] - 1 * hoop_pos[-1][2]
    x2 = hoop_pos[-1][0][0] + 1 * hoop_pos[-1][2]
    y1 = hoop_pos[-1][0][1] - 1 * hoop_pos[-1][3]
    y2 = hoop_pos[-1][0][1] + 0.5 * hoop_pos[-1][3]

    if x1 < x < x2 and y1 < y < y2:
        return True
    return False

#Remove inaccurate ball points
def clean_ball_pos(ball_pos, frame_count):
    #Remove inaccurate ball size
    if len(ball_pos) > 1:
        w1 = ball_pos[-2][2]
        h1 = ball_pos[-2][3]
        w2 = ball_pos[-1][2]
        h2 = ball_pos[-1][3]

        x1 = ball_pos[-2][0][0]
        y1 = ball_pos[-2][0][1]
        x2 = ball_pos[-1][0][0]
        y2 = ball_pos[-1][0][1]

        f1 = ball_pos[-2][1]
        f2 = ball_pos[-1][1]
        f_dif = f2 - f1

        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        max_dist = 4 * math.sqrt((w1) ** 2 + (h1) ** 2)

        #Ball should not move 4x its diameter within 5 frames
        if (dist > max_dist) and (f_dif < 5):
            ball_pos.pop()

        #Ball should be relatively square
        elif (w2*1.4 < h2) or (h2*1.4 < w2):
            ball_pos.pop()

    #Remove points older than 30 frames
    if len(ball_pos) > 0:
        if frame_count - ball_pos[0][1] > 30:
            ball_pos.pop(0)

    return ball_pos

#Remove inaccurate hoop points
def clean_hoop_pos(hoop_pos):
    #Prevent jumping between hoops
    if len(hoop_pos) > 1:
        x1 = hoop_pos[-2][0][0]
        y1 = hoop_pos[-2][0][1]
        x2 = hoop_pos[-1][0][0]
        y2 = hoop_pos[-1][0][1]

        w1 = hoop_pos[-2][2]
        h1 = hoop_pos[-2][3]
        w2 = hoop_pos[-1][2]
        h2 = hoop_pos[-1][3]

        f1 = hoop_pos[-2][1]
        f2 = hoop_pos[-1][1]

        f_dif = f2-f1

        dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)

        max_dist = 0.5 * math.sqrt(w1 ** 2 + h1 ** 2)

        #Hoop should not move 0.5x its diameter within 5 frames
        if dist > max_dist and f_dif < 5:
            hoop_pos.pop()

        #Hoop should be relatively square
        if (w2*1.3 < h2) or (h2*1.3 < w2):
            hoop_pos.pop()

    #Remove old points
    if len(hoop_pos) > 25:
        hoop_pos.pop(0)

    return hoop_pos

class shotCounter:
    def __init__(self):
        #Load YOLO model
        self.overlay_text = "Waiting..."
        self.model = YOLO("shot_counter.pt")
        
        #Uncomment to accelerate inference
        #self.model.half()
        
        self.class_names = ['Basketball', 'Basketball Hoop']
        self.device = get_device()
        #Uncomment to use webcam
        self.cap = cv2.VideoCapture("video_test_5.mp4")

        #Use video
        #self.cap = cv2.VideoCapture("video_test_5.mp4")

        self.ball_pos = []  #Ball positions
        self.hoop_pos = []  #Hoop positions

        self.frame_count = 0
        self.frame = None

        self.makes = 0
        self.attempts = 0

        #Detect shots (up and down regions)
        self.up = False
        self.down = False
        self.up_frame = 0
        self.down_frame = 0

        #Colors for make/miss
        self.fade_frames = 15
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)

        self.run()

    def run(self):
        while True:
            ret, self.frame = self.cap.read()

            if not ret:
                #End of video or error
                break

            results = self.model(self.frame, stream=True, device=self.device)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    #Bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1

                    #Confidence
                    conf = math.ceil((box.conf[0] * 100)) / 100

                    #Class Name
                    cls = int(box.cls[0])
                    current_class = self.class_names[cls]

                    center = (int(x1 + w / 2), int(y1 + h / 2))

                    #Create ball points if high confidence or near hoop
                    if (conf > .3 or (in_hoop_region(center, self.hoop_pos) and conf > 0.15)) and current_class == "Basketball":
                        self.ball_pos.append((center, self.frame_count, w, h, conf))
                        cv2.rectangle(self.frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), thickness=3)

                    #Create hoop points if high confidence
                    if conf > .5 and current_class == "Basketball Hoop":
                        self.hoop_pos.append((center, self.frame_count, w, h, conf))
                        cv2.rectangle(self.frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 255), thickness=2)

            self.clean_motion()
            self.shot_detection()
            self.display_score()
            self.frame_count += 1

            cv2.imshow('Frame', self.frame)

            #Close if 'q' is clicked
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def clean_motion(self):
        #Clean and display ball motion
        self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count)
        for i in range(0, len(self.ball_pos)):
            cv2.circle(self.frame, self.ball_pos[i][0], 2, (185, 0, 255), 3)

        #Clean hoop motion and display current hoop center
        if len(self.hoop_pos) > 1:
            self.hoop_pos = clean_hoop_pos(self.hoop_pos)
            cv2.circle(self.frame, self.hoop_pos[-1][0], 2, (128, 0, 0), 2)

    def shot_detection(self):
        if len(self.hoop_pos) > 0 and len(self.ball_pos) > 0:
            #Detect ball in 'up' and 'down' areas
            if not self.up:
                self.up = detect_up(self.ball_pos, self.hoop_pos)
                if self.up:
                    self.up_frame = self.ball_pos[-1][1]

            if self.up and not self.down:
                self.down = detect_down(self.ball_pos, self.hoop_pos)
                if self.down:
                    self.down_frame = self.ball_pos[-1][1]

            #Ball goes from 'up' to 'down', increase attempt and reset
            if self.frame_count % 10 == 0:
                if self.up and self.down and self.up_frame < self.down_frame:
                    self.attempts += 1
                    self.up = False
                    self.down = False

                    #If make, green overlay and "Make"
                    if score(self.ball_pos, self.hoop_pos):
                        self.makes += 1
                        self.overlay_color = (0, 255, 0)
                        self.overlay_text = "Make"
                        self.fade_counter = self.fade_frames

                    else:
                        self.overlay_color = (255, 0, 0)
                        self.overlay_text = "Miss"
                        self.fade_counter = self.fade_frames

    def display_score(self):
        #Add text
        text = str(self.makes) + " / " + str(self.attempts)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2

        cv2.putText(self.frame, text, (50, 505), font, font_scale, (255, 255, 255), thickness * 2)
        cv2.putText(self.frame, text, (50, 505), font, font_scale, (255, 255, 255), thickness)

        #Add overlay text for shot result
        if hasattr(self, 'overlay_text'):
            (text_width, text_height), _ = cv2.getTextSize(self.overlay_text, font, font_scale, thickness * 2)
            text_x = self.frame.shape[1] - text_width - 50
            text_y = 500

            cv2.putText(self.frame, self.overlay_text, (text_x, text_y), font, font_scale, self.overlay_color, thickness * 2)

        #Gradually fade out color after shot
        if self.fade_counter > 0:
            alpha = 0.2 * (self.fade_counter / self.fade_frames)
            self.frame = cv2.addWeighted(self.frame, 1 - alpha, np.full_like(self.frame, self.overlay_color), alpha, 0)
            self.fade_counter -= 1


if __name__ == "__main__":
    shotCounter()


#Traceback (most recent call last):
#File "C:\Users\ronit\Documents\Git\Capstone_4OI6\shot_counter\shot_count.py", line 334, in <module>
#shotCounter()
#File "C:\Users\ronit\Documents\Git\Capstone_4OI6\shot_counter\shot_count.py", line 212, in __init__
#self.run()
#File "C:\Users\ronit\Documents\Git\Capstone_4OI6\shot_counter\shot_count.py", line 224, in run
#for r in results:
#File "C:\Users\ronit\AppData\Local\Programs\Python\Python39\lib\site-packages\torch\utils\_contextlib.py", line 36, in generator_context
#response = gen.send(None)
#File "C:\Users\ronit\AppData\Local\Programs\Python\Python39\lib\site-packages\ultralytics\engine\predictor.py", line 261, in stream_inference
#preds = self.inference(im, *args, **kwargs)
#File "C:\Users\ronit\AppData\Local\Programs\Python\Python39\lib\site-packages\ultralytics\engine\predictor.py", line 145, in inference
#return self.model(im, augment=self.args.augment, visualize=visualize, embed=self.args.embed, *args, **kwargs)
#File "C:\Users\ronit\AppData\Local\Programs\Python\Python39\lib\site-packages\torch\nn\modules\module.py", line 1736, in _wrapped_call_impl
#return self._call_impl(*args, **kwargs)
#File "C:\Users\ronit\AppData\Local\Programs\Python\Python39\lib\site-packages\torch\nn\modules\module.py", line 1747, in _call_impl
#return forward_call(*args, **kwargs)
#File "C:\Users\ronit\AppData\Local\Programs\Python\Python39\lib\site-packages\ultralytics\nn\autobackend.py", line 555, in forward
#y = self.model(im, augment=augment, visualize=visualize, embed=embed)      
#File "C:\Users\ronit\AppData\Local\Programs\Python\Python39\lib\site-packages\torch\nn\modules\module.py", line 1736, in _wrapped_call_impl
#return self._call_impl(*args, **kwargs)
#File "C:\Users\ronit\AppData\Local\Programs\Python\Python39\lib\site-packages\torch\nn\modules\module.py", line 1747, in _call_impl
#return forward_call(*args, **kwargs)
#File "C:\Users\ronit\AppData\Local\Programs\Python\Python39\lib\site-packages\ultralytics\nn\tasks.py", line 109, in forward
#return self.predict(x, *args, **kwargs)
#File "C:\Users\ronit\AppData\Local\Programs\Python\Python39\lib\site-packages\ultralytics\nn\tasks.py", line 127, in predict
#return self._predict_once(x, profile, visualize, embed)
#\torch\nn\modules\module.py", line 1747, in _call_impl
#return forward_call(*args, **kwargs)
#File "C:\Users\ronit\AppData\Local\Programs\Python\Python39\lib\site-packages\ultralytics\nn\modules\conv.py", line 55, in forward_fuse  
#\torch\nn\modules\module.py", line 1747, in _call_impl
#return forward_call(*args, **kwargs)
#File "C:\Users\ronit\AppData\Local\Programs\Python\Python39\lib\site-packages\ultralytics\nn\modules\conv.py", line 55, in forward_fuse
#return self.act(self.conv(x))
#File "C:\Users\ronit\AppData\Local\Programs\Python\Python39\lib\site-packages\torch\nn\modules\module.py", line 1736, in _wrapped_call_impl
#return self._call_impl(*args, **kwargs)
#File "C:\Users\ronit\AppData\Local\Programs\Python\Python39\lib\site-packages\torch\nn\modules\module.py", line 1747, in _call_impl
#return forward_call(*args, **kwargs)
#File "C:\Users\ronit\AppData\Local\Programs\Python\Python39\lib\site-packages\torch\nn\modules\conv.py", line 554, in forward
#return self._conv_forward(input, self.weight, self.bias)
#File "C:\Users\ronit\AppData\Local\Programs\Python\Python39\lib\site-packages\torch\nn\modules\conv.py", line 549, in _conv_forward
#return F.conv2d(
#KeyboardInterrupt