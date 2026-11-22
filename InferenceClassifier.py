#This is the result of the training and data preparation. You can test the AI model here to see if it works
print("Importing libraries")

import cv2
import mediapipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pickle
import numpy
print("Done")


model_dict=pickle.load(open("C:/Users/USER/Desktop/Python Projects/TestingPythonVS/Computer vison/Sign Language detector/SignClassifiermodel.p","rb"))
model=model_dict["model"]
camera=cv2.VideoCapture(0)
FRAME_RATE=25
resultToLetter={0:"A",1:"B",2:"C"}
character=""


#This is a bunch of preparation for handlandmarker stuff just don't tough it
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
HandConnections = vision.HandLandmarksConnections
drawing = vision.drawing_utils
draw_styles = vision.drawing_styles
# 2. Initializes the hand detector. Yes it is sitll using an absolute file path as I REALLY do not trust my computer. Change it to Sign Language detector/hand_landmarker.task if necessary.
base_options = python.BaseOptions(model_asset_path="C:/Users/USER/Desktop/Python Projects/TestingPythonVS/Computer vison/Sign Language detector/hand_landmarker.task")
options = HandLandmarkerOptions(
    base_options=base_options,  
    num_hands=2
)
detector = vision.HandLandmarker.create_from_options(options)




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

            #This loop saves the data into an array to be used by the model. Yes it is a global variable tht 'm just calling insde a subprogram
            for mark in hand_landmarks:
                x_coord=mark.x
                y_coord=mark.y
                hand_data.append(x_coord)
                hand_data.append(y_coord)



    return annotated_image


while True:
    hand_data=[]
    read,frame_bgr=camera.read()
    frame_rgb=cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB)
    mp_image=mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=frame_rgb)
    figure=detector.detect(mp_image)
    output_rgb=landmark(figure,frame_rgb)
    output_bgr=cv2.cvtColor(output_rgb,cv2.COLOR_RGB2BGR)
    if len(hand_data)!=0:
        prediction=model.predict([numpy.asarray(hand_data)])
        print(prediction)
        character=resultToLetter[int(prediction[0])]
        print(f"Predicted Character:{character}")
    else:
        character="Nothing"
    cv2.putText(output_bgr, f"Character: {character}", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)

    cv2.imshow("Window",output_bgr)

    if cv2.waitKey(FRAME_RATE) & 0xFF==ord("z"):
        break


camera.release()
cv2.destroyAllWindows()