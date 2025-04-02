from ultralytics import YOLO

model = YOLO("ball_rimV8.pt")
print(model.names)  # This will print the dictionary of class_id -> class_name
