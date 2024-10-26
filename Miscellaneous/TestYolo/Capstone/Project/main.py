# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import cv2
import os

def ballImageDetction():

    # Define the path to the images directory
    images_directory = r'C:\Users\antho\OneDrive\Desktop\Capstone\TestImages'

    # Load the YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Iterate over each image in the directory
    for image_name in os.listdir(images_directory):
        if image_name.endswith('.jpg'):
            image_path = os.path.join(images_directory, image_name)
            image = cv2.imread(image_path)

            # Perform inference
            results = model(image)

            # Filter results to find basketball class (if available)
            # Check available classes using `results.names` and match with the class index
            basketball_class_index = [k for k, v in results.names.items() if 'basketball' in v.lower()]

            if basketball_class_index:
                filtered_results = results.xyxy[0][results.xyxy[0][:, -1] == basketball_class_index[0]]
                print(f"Results for {image_name}:")
                print(filtered_results)

            # Display results
            results.show()

            # Save the image with detections (optional)
            results.save(save_dir=os.path.join(images_directory, 'detections'))

            # Show the image using OpenCV (if you want to visualize in a window)
            # cv2.imshow('Detection', image)
            # cv2.waitKey(0)

    # Close all windows if using OpenCV
    cv2.destroyAllWindows()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ballImageDetction()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
