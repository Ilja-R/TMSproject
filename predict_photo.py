import os

from ultralytics import YOLO
import cv2


def preprocess_image(input_image):
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    # perform otsu thresh (using binary inverse since opencv contours work better with white text)
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # apply dilation
    dilation = cv2.dilate(thresh, rect_kern, iterations=1)
    return dilation


PHOTOS_DIR = os.path.join('.', 'photos')

photo_path = os.path.join(PHOTOS_DIR, 'IMG_3227.JPG')

image = cv2.imread(photo_path)

model_path = os.path.join('.', 'runs', 'detect', 'train2', 'weights', 'last.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5
# numbers_threshold = 0.2

photo_path_out = os.path.join(".", "photos", "out_2.png")

results = model(image)[0]

for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score > threshold:
        # create new image only of the detected objects
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Crop the bounding box region from the image
        cropped_image = image[y1:y2, x1:x2]

        preprocessed_image = preprocess_image(cropped_image)

        cv2.imwrite(photo_path_out, preprocessed_image)
