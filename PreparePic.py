import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
print("This is it: ",os.getcwd())

# 1. Set up aliases (replaces mp.solutions.*)

HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
HandConnections = vision.HandLandmarksConnections
drawing = vision.drawing_utils
draw_styles = vision.drawing_styles

# 2. Configure and initialize the detector (replaces landmark_hands.Hands)
base_options = python.BaseOptions(model_asset_path="C:/Users/USER/Desktop/Python Projects/TestingPythonVS/Computer vison/hand_landmarker.task")
options = HandLandmarkerOptions(
    base_options=base_options,  
    num_hands=2
)

detector = vision.HandLandmarker.create_from_options(options)

# 3. Equivalent to your landmark() drawing function
def landmark(detection_result, img_rgb):
    annotated_image = img_rgb.copy()
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                HandConnections.HAND_CONNECTIONS,
                draw_styles.get_default_hand_landmarks_style(),
                draw_styles.get_default_hand_connections_style()
            )
    return annotated_image

# --- Example Usage ---
# Load an image with OpenCV and convert BGR to RGB
image_path=""
bgr_image = cv2.imread("C:/Users/USER/Desktop/Python Projects/TestingPythonVS/Computer vison/Sign Language detector/myHand.jpg")
img_rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

# Wrap the RGB numpy array in a MediaPipe Image object and detect
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

#This is to create the object 
figure = detector.detect(mp_image)

# Draw landmarks and convert back to BGR if displaying/saving with OpenCV
output_rgb = landmark(figure, img_rgb)
output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
cv2.imshow("window",output_bgr)
cv2.waitKey(0)