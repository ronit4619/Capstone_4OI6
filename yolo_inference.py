from ultralytics import YOLO 

model = YOLO('models/basketball.pt')

results = model.predict('input_videos/vid2.mp4',save=True)
print(results[0])
print('=====================================')
for box in results[0].boxes:
    print(box)