import os
import cv2

from ultralytics import YOLO

model_path = os.path.join('.', 'runs', 'detect', 'train2', 'weights', 'last.pt')

# Load a model
MODEL = YOLO(model_path)  # load a custom model

THRESHOLD = 0.5

def preprocess_image(input_image):
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    # perform otsu thresh (using binary inverse since opencv contours work better with white text)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # apply dilation (makes the image thicker)
    dilation = cv2.dilate(thresh, rect_kern, iterations=1)
    return dilation

def get_scale_area(input_image):
    results = MODEL(input_image)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > THRESHOLD:
            # create new image only of the detected objects
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Crop the bounding box region from the image
            cropped_image = input_image[y1:y2, x1:x2]

            scale_area = preprocess_image(cropped_image)

            return scale_area

    return None
