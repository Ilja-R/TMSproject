from google.cloud import vision
import cv2

try:
    # Need to have GOOGLE api set up to use this
    client = vision.ImageAnnotatorClient()
except Exception as e:
    print("Error initializing the Vision API client:", e)
    client = None

def detect_text(cv2_image):

    if client is None:
        print("Vision API client not initialized.")
        return "Could not set up the Vision API client."

    # Encode the image as a JPEG or PNG and read the binary data
    _, encoded_image = cv2.imencode('.png', cv2_image)
    content = encoded_image.tobytes()

    # Create an Image object for the Vision API with the binary content
    image = vision.Image(content=content)

    # Set image context with language hints (e.g., 'en' for English)
    image_context = vision.ImageContext(language_hints=['en'])

    response = client.text_detection(image=image, image_context=image_context)
    texts = response.text_annotations

    if texts:
        # Print all detected text (the first element contains the full text)
        print(f"Detected text: {texts[0].description}")
        return texts[0].description
    else:
        print("No text detected in the image.")
        return None
