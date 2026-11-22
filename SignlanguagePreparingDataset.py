#This is extra important as it prepares the data into a form that the model can understand. I would write it to a pickle file at the end
#Libraries
print("Importing libraries")
import os
import mediapipe
import cv2
import pickle
import matplotlib.pyplot as pplot
from mediapipe.tasks import python
from mediapipe.tasks.python import vision



print("Done Importing")

#Legacy Code just ignore this
#def landmark(figure):
#    if figure.multi_hand_landmarks:
 #       for hand_landmarks in figure.multi_hand_landmarks:
  ##             img_rgb, 
   # image to draw
    #            hand_landmarks,  # model output
#              landmark_hands.HAND_CONNECTIONS,  # hand connections
 #               draw_styles.get_default_hand_landmarks_style(),#Drawing styles
  #              draw_styles.get_default_hand_connections_style())


#Subprograms
#This function is very important as it does the main work of marking the hands for the AI
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
                data_addon.append(x_coord)
                data_addon.append(y_coord)



    return annotated_image


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




DATA="C:/Users/USER/Desktop/Python Projects/TestingPythonVS/Computer vison/Sign Language detector/trainingdata"

#This is just a tes image I used, you can change it to anything you want. I used myHand.jpg
SmallSample="C:/Users/USER/Desktop/Python Projects/TestingPythonVS/Computer vison/Sign Language detector/myHand.jpg"

#Save file
SaveFile="C:/Users/USER/Desktop/Python Projects/TestingPythonVS/Computer vison/Sign Language detector/BinaryData.pickle"

#Legacy code-Ignore or just delete this.
#Detecting landmarks using mediapipe
#landmark_hands=mediapipe.solutions.hands
#drawing=mediapipe.solutions.drawing_utils
#draw_styles=mediapipe.solutions.drawing_styles
#hands=landmark_hands.Hands(static_image_mode=True, min_detection_confidnce=0.3)


trainingData=[]
nameList=[]

#Main Loop
for dir_ in os.listdir(DATA):
    letter=os.listdir(os.path.join(DATA,dir_))
   
   
    for img_path in letter:
        data_addon=[]

        img_bgr = cv2.imread(os.path.join(DATA,dir_,img_path))#You can use SmallSample here
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        #Swapping between rgb and bgr cause cv2 uses bgr by default and the other use rgb
        # Wrap the RGB numpy array in a MediaPipe Image object and detect
        mp_image = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=img_rgb)
        #This is to create the object 
        figure = detector.detect(mp_image)
        output_rgb = landmark(figure, img_rgb)#Call to my created function
        output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)

        #Adding data to the arrays
        trainingData.append(data_addon)
        nameList.append(dir_)

 

#It is then saved to a pickle file. I think I can convert it to csv but I'll worry about that later.
with open(SaveFile,"wb") as myFile:
    pickle.dump({"data":trainingData,"Label":nameList},myFile)




